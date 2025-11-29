#include <chrono>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <zlib.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <memory>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include "libstemmer.h"

// NOTE: build and read on same architecture (endianness, size of types)

namespace fs = std::filesystem;

// constants for memory size estimation (64 bit system)
const size_t MAP_NODE_OVERHEAD = 32; 
const size_t VECTOR_OVERHEAD = 24;

struct DocInfo {
    std::string url;
    std::string title;
};

struct ParsedDoc {
    uint32_t doc_id;
    std::string url;
    std::string title;
    std::string body;
};

class DocStoreWriter {
private:
    std::ofstream out_stream;     
    std::ofstream offset_stream;
    uint64_t current_byte_offset; // offset where next doc will be written/read
    uint32_t doc_count;

public:
    void init(const std::string& filename_base) {
        out_stream.open(filename_base + ".docstore", std::ios::binary | std::ios::out | std::ios::trunc);
        offset_stream.open(filename_base + ".docstore_offsets", std::ios::binary | std::ios::out | std::ios::trunc);
        
        current_byte_offset = 0; 
        doc_count = 0;
 
        out_stream.write(reinterpret_cast<const char*>(&doc_count), sizeof(doc_count));
        current_byte_offset += sizeof(doc_count);
    }

    void add_document(uint32_t doc_id, const std::string& url, const std::string& title) {
        /*
        offset_stream:
        0   (offset of doc_id 0)
        18  (offset of doc_id 1)
        ...

        out_stream:
        [0-3]   url_len = 5
        [4-8]   'a' '.' 'c' 'o' 'm'
        [9-12]  title_len = 5
        [13-17] 'H' 'e' 'l' 'l' 'o'

        [18-21] url_len = 11
        [22-32] 'e' 'x' 'a' 'm' 'p' 'l' 'e' '.' 'o' 'r' 'g'
        [33-34] title_len = 2
        [35-36] 'H' 'i'
        */
        offset_stream.write(reinterpret_cast<const char*>(&current_byte_offset), sizeof(current_byte_offset));

        uint32_t url_len = url.size();
        out_stream.write(reinterpret_cast<const char*>(&url_len), sizeof(url_len));
        out_stream.write(url.data(), url_len);

        uint32_t title_len = title.size();
        out_stream.write(reinterpret_cast<const char*>(&title_len), sizeof(title_len));
        out_stream.write(title.data(), title_len);

        // faster than tellp()
        current_byte_offset += sizeof(uint32_t) + url_len + sizeof(uint32_t) + title_len;
        
        doc_count++;
    }

    void close() {
        if (out_stream.is_open()) {
            // write actual doc count at the beginning of the data file which was 0 before
            out_stream.seekp(0);
            out_stream.write(reinterpret_cast<const char*>(&doc_count), sizeof(doc_count));
            out_stream.close();
            offset_stream.close();
        }
    }
};

// NOTE: Snowball stemmer instance is not thread-safe
struct SnowballStemmer {
    struct sb_stemmer* stemmer;
    SnowballStemmer() {
        stemmer = sb_stemmer_new("english", nullptr);
    }
    ~SnowballStemmer() {
        sb_stemmer_delete(stemmer);
    }
    std::string stem(const std::string& word) {
        const sb_symbol* stemmed = sb_stemmer_stem(stemmer,
            reinterpret_cast<const sb_symbol*>(word.c_str()), word.size());
        int out_len = sb_stemmer_length(stemmer);
        if (stemmed == nullptr || out_len <= 0) return std::string();
        return std::string(reinterpret_cast<const char*>(stemmed), static_cast<size_t>(out_len));
    }
};
SnowballStemmer stemmer;

struct PostingList {
    std::vector<uint32_t> postings;
    std::vector<uint32_t> term_frequencies;
    std::vector<std::vector<uint32_t>> positions;
    std::unordered_map<uint32_t, uint32_t> skip_pointers;
    
    size_t add_document_occurrences(uint32_t doc_id, const std::vector<uint32_t>& new_positions) {
        size_t memory_delta = 0;

        if (!postings.empty() && postings.back() == doc_id) {
            // existing last doc: extend its positions vector
            term_frequencies.back() += new_positions.size();
            auto& pos_vec = positions.back();

            size_t old_cap = pos_vec.capacity();
            pos_vec.insert(pos_vec.end(), new_positions.begin(), new_positions.end());
            size_t new_cap = pos_vec.capacity();

            memory_delta += (new_cap - old_cap) * sizeof(uint32_t);
        } else {
            // new doc entry
            postings.push_back(doc_id);
            term_frequencies.push_back(new_positions.size());

            positions.emplace_back();
            auto& pos_vec = positions.back();
            pos_vec.reserve(new_positions.size());
            pos_vec.insert(pos_vec.end(), new_positions.begin(), new_positions.end());

            memory_delta += sizeof(uint32_t) * 2; // doc_id + tf (stored elsewhere)
            memory_delta += VECTOR_OVERHEAD; // vector structure overhead
            memory_delta += pos_vec.capacity() * sizeof(uint32_t);
        }

        return memory_delta;
    }

    void build_skip_pointers() {
        if (postings.empty()) return;
        size_t skip_interval = static_cast<size_t>(std::sqrt(postings.size()));
        if (skip_interval < 2) return; 

        skip_pointers.clear();
        for (size_t i = 0; i + skip_interval < postings.size(); i += skip_interval) {
            skip_pointers[i] = i + skip_interval;
        }
    }
    
    static PostingList merge(const PostingList& a, const PostingList& b) {
        PostingList result;
        size_t i = 0, j = 0;
        
        // reserve memory to avoid reallocations
        result.postings.reserve(a.postings.size() + b.postings.size());
        result.term_frequencies.reserve(a.term_frequencies.size() + b.term_frequencies.size());
        result.positions.reserve(a.positions.size() + b.positions.size());

        while (i < a.postings.size() || j < b.postings.size()) {
            
            uint32_t doc_id_a = (i < a.postings.size()) ? a.postings[i] : UINT32_MAX;
            uint32_t doc_id_b = (j < b.postings.size()) ? b.postings[j] : UINT32_MAX;

            if (doc_id_a < doc_id_b) {
                result.postings.push_back(doc_id_a);
                result.term_frequencies.push_back(a.term_frequencies[i]);
                result.positions.push_back(a.positions[i]);
                i++;
            } else if (doc_id_b < doc_id_a) {
                result.postings.push_back(doc_id_b);
                result.term_frequencies.push_back(b.term_frequencies[j]);
                result.positions.push_back(b.positions[j]);
                j++;
            } else if (doc_id_a != UINT32_MAX) {
                uint32_t doc_id = doc_id_a;
                result.postings.push_back(doc_id);
                
                result.term_frequencies.push_back(a.term_frequencies[i] + b.term_frequencies[j]);
                
                std::vector<uint32_t> pos_a = a.positions[i];
                const auto& pos_b = b.positions[j];
                pos_a.insert(pos_a.end(), pos_b.begin(), pos_b.end());
                result.positions.push_back(std::move(pos_a));

                i++;
                j++;
            } else {
                // both UINT32_MAX -> end
                break;
            }
        }
                
        return result;
    }
    
    // estimate total memory size used for dumping logic
    size_t memory_size() const {
        size_t size = 0;
        
        size += postings.capacity() * sizeof(uint32_t);
        size += term_frequencies.capacity() * sizeof(uint32_t);
        
        size += positions.capacity() * VECTOR_OVERHEAD;
        
        for (const auto& pos : positions) {
            size += pos.capacity() * sizeof(uint32_t); 
        }
        
        size += skip_pointers.size() * (MAP_NODE_OVERHEAD + sizeof(uint32_t) * 2);
        
        return size;
    }
};

class MMapReader {
public:
    const char* data;
    size_t size;
    int fd;

    MMapReader(const std::string& filename) {
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Could not open file for mmap");

        struct stat sb;
        if (fstat(fd, &sb) == -1) throw std::runtime_error("Could not stat file");
        size = sb.st_size;

        if (size == 0) {
            data = nullptr;
            return;
        }

        void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("mmap failed");

        data = static_cast<const char*>(mapped);
        madvise(mapped, size, MADV_SEQUENTIAL);
    }

    ~MMapReader() {
        if (data) munmap(const_cast<char*>(data), size);
        if (fd != -1) close(fd);
    }
    
    MMapReader(const MMapReader&) = delete;
    MMapReader& operator=(const MMapReader&) = delete;
};

template <typename T>
T read_val(const char*& ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

uint64_t write_posting_list(std::ofstream& out, const PostingList& pl, bool with_skip_pointers = false) {
    /*
    with_skip_pointers: we don't need skip pointers for temp spilling to disk, only for final index

    Example: 
    pl:
        postings = [3, 7]
        term_frequencies = {3: 4, 7: 1}
        positions = {
            3: [0, 5, 9, 20],
            7: [13]
        }

    In binary file:
    count_docs = 2

    3 (first doc_id)
    4 (tf for doc_id 3)
    4 (pos_count for doc_id 3)
    [0, 5, 9, 20] (positions for doc_id 3)

    7 (second doc_id)
    1 (tf for doc_id 7)
    1 (pos_count for doc_id 7)
    [13] (positions for doc_id 7)
    */

    uint64_t offset = out.tellp();
    uint32_t count_docs = pl.postings.size();
    out.write(reinterpret_cast<const char*>(&count_docs), sizeof(count_docs));
    
    for (size_t i = 0; i < count_docs; ++i) {
        uint32_t doc_id = pl.postings[i];
        uint32_t tf = pl.term_frequencies[i];
        const auto& pos = pl.positions[i];
        
        out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));
        out.write(reinterpret_cast<const char*>(&tf), sizeof(tf));
        
        uint32_t pos_count = pos.size();
        out.write(reinterpret_cast<const char*>(&pos_count), sizeof(pos_count));
        out.write(reinterpret_cast<const char*>(pos.data()), pos_count * sizeof(uint32_t));
    }
    
    if (with_skip_pointers) {
        uint32_t skip_count = pl.skip_pointers.size();
        out.write(reinterpret_cast<const char*>(&skip_count), sizeof(skip_count));
        for (const auto& [from_idx, to_idx] : pl.skip_pointers) {
            out.write(reinterpret_cast<const char*>(&from_idx), sizeof(from_idx));
            out.write(reinterpret_cast<const char*>(&to_idx), sizeof(to_idx));
        }
    }
    
    return offset;
}

PostingList read_posting_list(const char*& ptr) {
    PostingList pl;
    
    uint32_t count_docs = read_val<uint32_t>(ptr);
    pl.postings.reserve(count_docs);
    pl.term_frequencies.reserve(count_docs);
    pl.positions.reserve(count_docs);
    
    for (uint32_t i = 0; i < count_docs; i++) {
        uint32_t doc_id = read_val<uint32_t>(ptr);
        uint32_t tf = read_val<uint32_t>(ptr);
        uint32_t pos_count = read_val<uint32_t>(ptr);
        
        pl.postings.push_back(doc_id);
        pl.term_frequencies.push_back(tf);
        
        std::vector<uint32_t> positions(pos_count);
        std::memcpy(positions.data(), ptr, pos_count * sizeof(uint32_t));
        ptr += pos_count * sizeof(uint32_t);
        
        pl.positions.push_back(std::move(positions));
    }
    
    return pl;
}

struct MergeState {
    std::string term;
    const char* current_ptr;
    int file_index;

     // std::priority_queue is max-heap by default, so we invert the comparison
    bool operator>(const MergeState& other) const {
        return term > other.term;
    }
};

class InvertedIndexBuilder {
private:
    std::unordered_map<std::string, uint32_t> term_to_index;
    std::vector<PostingList> posting_lists;
    size_t current_memory;
    size_t memory_limit;
    std::vector<std::string> spilled_files;
    int spill_counter;
    std::string temp_dir;
    DocStoreWriter doc_store; 
    std::string temp_docstore_base;

public:
    InvertedIndexBuilder(size_t mem_limit_mb = 1024) 
        : current_memory(0), 
          memory_limit(mem_limit_mb * 1024 * 1024),
          spill_counter(0),
          temp_dir("temp_index") {
        fs::create_directories(temp_dir);
        // writing docstore to temp location instead of final location in case of crashes
        temp_docstore_base = temp_dir + "/temp_docstore";
        doc_store.init(temp_docstore_base);
    }
    
    void add_document(uint32_t doc_id, const std::vector<std::string>& tokens,
                      const std::string& url = "", const std::string& title = "") {
        std::unordered_map<std::string, std::vector<uint32_t>> term_positions;
        for (uint32_t pos = 0; pos < tokens.size(); pos++) {
            term_positions[tokens[pos]].push_back(pos);
        }
        
        for (const auto& [term, positions] : term_positions) {
            uint32_t pl_index;
            bool is_new_term = false;
            
            auto it = term_to_index.find(term);
            if (it == term_to_index.end()) {
                is_new_term = true;
                pl_index = posting_lists.size();
                term_to_index[term] = pl_index;
                posting_lists.emplace_back();
            } else {
                pl_index = it->second;
            }
            
            auto& pl = posting_lists[pl_index];
            size_t bytes_added = pl.add_document_occurrences(doc_id, positions);
            
            if (is_new_term) {
                bytes_added += MAP_NODE_OVERHEAD + term.size() + sizeof(PostingList);
            }
            
            current_memory += bytes_added;
        }
        
        if (current_memory > memory_limit) {
            spill_to_disk();
        }
        
        doc_store.add_document(doc_id, url, title);
    }
    
    void spill_to_disk() {
        /*
        * Example binary file layout for spill_to_disk() with 2 terms: "apple" and "banana"
        *
        * Example:
        * - Term "apple" appears in doc_id 3 (tf=4, positions=[0,5]) and doc_id 7 (tf=2, positions=[10])
        * - Term "banana" appears in doc_id 2 (tf=1, positions=[1])
        *
        * Byte-level layout (offsets in bytes):
        *
        * [0-3]   : term_count = 2 (uint32_t)
        *
        * Term 1: "apple"
        * [4-7]   : term_len = 5 (uint32_t)
        * [8-12]  : 'a' 'p' 'p' 'l' 'e'           (5 bytes)
        * [13-16] : posting_count = 2 (uint32_t)
        *   Doc 1:
        *   [17-20] : doc_id = 3
        *   [21-24] : tf = 4
        *   [25-28] : pos_count = 2
        *   [29-36] : positions = 0, 5 (2 * 4 bytes)
        *   Doc 2:
        *   [37-40] : doc_id = 7
        *   [41-44] : tf = 2
        *   [45-48] : pos_count = 1
        *   [49-52] : positions = 10
        * [53-56] : skip_count = 0 (no skip pointers)
        *
        * Term 2: "banana"
        * [57-60] : term_len = 6 (uint32_t)
        * [61-66] : 'b' 'a' 'n' 'a' 'n' 'a'       (6 bytes)
        * [67-70] : posting_count = 1 (uint32_t)
        *   Doc 1:
        *   [71-74] : doc_id = 2
        *   [75-78] : tf = 1
        *   [79-82] : pos_count = 1
        *   [83-86] : positions = 1
        * [87-90] : skip_count = 0
        */

        std::string filename = temp_dir + "/spill_" + std::to_string(spill_counter++) + ".bin";
        std::ofstream out(filename, std::ios::binary);
        
        std::vector<std::pair<std::string, uint32_t>> sorted_terms;
        sorted_terms.reserve(term_to_index.size());
        
        for (const auto& [term, idx] : term_to_index) {
            sorted_terms.emplace_back(term, idx);
        }
        
        std::sort(sorted_terms.begin(), sorted_terms.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        
        uint32_t term_count = sorted_terms.size();
        out.write(reinterpret_cast<const char*>(&term_count), sizeof(term_count));
        
        for (const auto& [term, idx] : sorted_terms) {
            uint32_t term_len = term.size();
            out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            out.write(term.data(), term_len);
            
            const PostingList& pl = posting_lists[idx];
            write_posting_list(out, pl);
        }
        
        out.close();
        spilled_files.push_back(filename);
        
        std::cout << "Spilled " << term_to_index.size() << " terms to " << filename 
                << " (" << current_memory / (1024*1024) << " MB)" << std::endl;
        
        term_to_index.clear();
        posting_lists.clear();
        current_memory = 0;
    }
    
    void finalize(const std::string& output_file_base) {
        if (!term_to_index.empty()) {
            spill_to_disk();
        }

        doc_store.close();
        
        // move docstore files to final location
        if (fs::exists(output_file_base + ".docstore")) fs::remove(output_file_base + ".docstore");
        if (fs::exists(output_file_base + ".docstore_offsets")) fs::remove(output_file_base + ".docstore_offsets");
        fs::rename(temp_docstore_base + ".docstore", output_file_base + ".docstore");
        fs::rename(temp_docstore_base + ".docstore_offsets", output_file_base + ".docstore_offsets");

        std::cout << "Merging " << spilled_files.size() << " spilled files using Priority Queue..." << std::endl;
        auto merge_start = std::chrono::high_resolution_clock::now();

        std::vector<std::unique_ptr<MMapReader>> readers;
        std::vector<const char*> file_ptrs;  // track current position per file

        // queue is max-heap by default, implement min-heap by inverting comparison with MergeState::operator>
        std::priority_queue<MergeState, std::vector<MergeState>, std::greater<MergeState>> merge_queue;

        // open all temp files and push the FIRST term of each file into the queue
        // queue contains term, offset in file, file index
        // result: smallest term per file, so if e.g. "and" is smallest for 8/10 files, "or" for 2 files,
        // we have 8 entries with "and" and 2 with "or" in the queue
        for (size_t i = 0; i < spilled_files.size(); ++i) {
            readers.push_back(std::make_unique<MMapReader>(spilled_files[i]));
            
            const char* ptr = readers.back()->data; // start of file
            const char* end = ptr + readers.back()->size; // end of file

            // every spill file starts with uint32_t term_count
            // --> check that not empty
            if (ptr + sizeof(uint32_t) <= end) {
                uint32_t term_count = read_val<uint32_t>(ptr);
                
                if (term_count > 0 && ptr + sizeof(uint32_t) <= end) {
                    // read first term
                    uint32_t term_len = read_val<uint32_t>(ptr);
                    if (ptr + term_len <= end) {
                        std::string term(ptr, term_len);
                        ptr += term_len;

                        merge_queue.push({term, ptr, (int)i});
                        file_ptrs.push_back(ptr);
                        continue;
                    }
                }
            }

            // if file is empty or malformed, track it as exhausted
            file_ptrs.push_back(nullptr);
        }
        
        // merging process
        std::ofstream postings_out(output_file_base + ".postinglists", std::ios::binary);
        std::ofstream index_out(output_file_base + ".index", std::ios::binary);
        
        uint64_t term_counter = 0;
        
        while (!merge_queue.empty()) {
            MergeState min_state = merge_queue.top();
            merge_queue.pop();
            std::string min_term = min_state.term;
            
            term_counter++;
            if (term_counter % 100000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - merge_start).count();
                std::cout << "Merged " << term_counter << " terms..."
                        << " | elapsed time: " << elapsed << " s" << std::endl;
            }
            
            PostingList merged_pl;

            do {
                int file_idx = min_state.file_index;
                
                const char* ptr = min_state.current_ptr;
                
                PostingList pl = read_posting_list(ptr);

                if (merged_pl.postings.empty()) merged_pl = std::move(pl);
                else merged_pl = PostingList::merge(merged_pl, pl);

                const char* end = readers[file_idx]->data + readers[file_idx]->size;

                // after every postinglist comes next term length (uint32_t) + term if there is a next term
                if (ptr + sizeof(uint32_t) <= end) {
                    uint32_t term_len;
                    std::memcpy(&term_len, ptr, sizeof(uint32_t));
                    
                    if (ptr + sizeof(uint32_t) + term_len <= end) {
                        
                        ptr += sizeof(uint32_t);
                        std::string next_term(ptr, term_len);
                        ptr += term_len;
                        
                        merge_queue.push({next_term, ptr, file_idx});
                    }
                }

                if (merge_queue.empty() || merge_queue.top().term != min_term) break;

                min_state = merge_queue.top();
                merge_queue.pop();

            } while (true);

            merged_pl.build_skip_pointers();
            uint64_t offset = write_posting_list(postings_out, merged_pl, true);
            
            uint32_t term_len = min_term.size();
            index_out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            index_out.write(min_term.data(), term_len);
            index_out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        }

        postings_out.close();
        index_out.close();
        readers.clear();
   
        auto merge_end = std::chrono::high_resolution_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(merge_end - merge_start).count();
        std::cout << "Merging finished. Total terms: " << term_counter 
                << ", elapsed time: " << total_elapsed << "s" << std::endl;
        
        std::cout << "Cleaning up temporary files..." << std::endl;
        for (const auto& filename : spilled_files) fs::remove(filename);
        fs::remove(temp_dir);
        std::cout << "Index built successfully: " << output_file_base << std::endl;
    }
};

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;

    for (char c : text) {
        if (std::isalnum(c)) {
            token += std::tolower(c);
        } else if (!token.empty()) {
            tokens.push_back(stemmer.stem(token));
            token.clear();
        }
    }
    if (!token.empty()) {
        tokens.push_back(stemmer.stem(token));
    }
    return tokens;
}

uint32_t next_doc_id = 0;

ParsedDoc parse_line(const std::string& line) {
    size_t first_tab = line.find('\t');
    size_t second_tab = line.find('\t', first_tab + 1);
    size_t third_tab = line.find('\t', second_tab + 1);

    ParsedDoc doc;
    std::string docid_str = line.substr(0, first_tab);
    doc.doc_id = next_doc_id++;
    doc.url = line.substr(first_tab + 1, second_tab - first_tab - 1);
    doc.title = line.substr(second_tab + 1, third_tab - second_tab - 1);
    doc.body = line.substr(third_tab + 1);
    return doc;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <memory-limit>" << std::endl;
        return 1;
    }

    std::cout << "Starting index building with memory limit: " << argv[1] << " MB" << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();

    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path project_root = exe_path.parent_path().parent_path();
    
    std::string data_dir_str = "data";
    
    const char* test_env = std::getenv("ENV"); // for integration tests, test with controlled and small dataset in test_data
    if (test_env && std::string(test_env) == "TEST_ENV") {
        std::cout << "TEST ENVIRONMENT, building index with test data." << std::endl;
        data_dir_str = "test_data";
    } 

    std::filesystem::path input_file = project_root / data_dir_str / "msmarco.tsv.gz";
    std::filesystem::path output_dir = project_root.parent_path() / "index" / "bin"; // put in parallel directory index/ where python code expects it
  
    InvertedIndexBuilder builder(std::stoull(argv[1]));

    gzFile in = gzopen(input_file.string().c_str(), "rb");
    if (!in) {
        std::cerr << "Failed to open msmarco.tsv.gz" << std::endl;
        return 1;
    }

    const size_t CHUNK_SIZE = 4 * 1024 * 1024;
    std::vector<char> decompressed_data(CHUNK_SIZE + 1, 0);
    std::string line_buffer;

    uint32_t doc_count = 0;
    auto start_read = std::chrono::high_resolution_clock::now();
    auto start_chunk = start_read;

    uint32_t max_docs = -1; // set to a positive number for testing with limited docs
    while (true) {
        int bytes_read = gzread(in, decompressed_data.data(), CHUNK_SIZE);
        if (bytes_read <= 0) break;

        decompressed_data[bytes_read] = 0; // null terminator
        line_buffer.append(decompressed_data.data(), bytes_read);

        size_t start = 0;
        for (size_t pos; (pos = line_buffer.find('\n', start)) != std::string::npos; start = pos + 1) {
            std::string_view line(&line_buffer[start], pos - start);
            while (!line.empty() && line.back() == '\r') line.remove_suffix(1);

            ParsedDoc doc = parse_line(std::string(line));
            std::vector<std::string> tokens = tokenize(doc.body);
            builder.add_document(doc.doc_id, tokens, doc.url, doc.title);

            doc_count++;
            if (doc_count % 10000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double chunk_sec = std::chrono::duration<double>(now - start_chunk).count();
                double total_sec = std::chrono::duration<double>(now - start_read).count();
                double docs_per_sec = doc_count / total_sec;
                std::cout << "Processed " << doc_count << " docs | "
                        << "Chunk: " << chunk_sec << "s, Total: " << total_sec << "s | "
                        << "Speed: " << (int)docs_per_sec << " docs/s" << std::endl;
                start_chunk = now;
            }
            if (max_docs != -1 && doc_count >= max_docs) break;
        }

        line_buffer = line_buffer.substr(start);
        if (max_docs != -1 && doc_count >= max_docs) break;
    }

    // process last line if no \n at end
    if (doc_count < max_docs && !line_buffer.empty()) {
        ParsedDoc doc = parse_line(line_buffer);
        std::vector<std::string> tokens = tokenize(doc.body);
        builder.add_document(doc.doc_id, tokens, doc.url, doc.title);
    }

    gzclose(in);
    
    auto end_read = std::chrono::high_resolution_clock::now();
    std::cout << "Finished reading. Time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_read - start_read).count()
              << "s" << std::endl;

    std::filesystem::create_directories(output_dir);

    builder.finalize((output_dir / "inverted_index").string());

    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total).count()
              << "s" << std::endl;

    return 0;
}
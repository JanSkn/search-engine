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
#include "libstemmer.h"

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
    std::unordered_map<uint32_t, uint32_t> term_frequencies;
    std::unordered_map<uint32_t, std::vector<uint32_t>> positions;
    std::unordered_map<uint32_t, uint32_t> skip_pointers;
    
    size_t add_document_occurrences(uint32_t doc_id, const std::vector<uint32_t>& new_positions) {
        size_t memory_delta = 0;

        if (term_frequencies.find(doc_id) == term_frequencies.end()) {
            // document does not exist yet, initialize
            postings.push_back(doc_id);
            term_frequencies[doc_id] = 0;
            
            memory_delta += sizeof(uint32_t); // postings vec
            memory_delta += MAP_NODE_OVERHEAD + sizeof(uint32_t) * 2; // TF map node (key+val)
            memory_delta += MAP_NODE_OVERHEAD + sizeof(uint32_t) + VECTOR_OVERHEAD; // pos map node + vec struct
        }

        term_frequencies[doc_id] += new_positions.size();
        
        auto& pos_vec = positions[doc_id];
        if (pos_vec.empty()) {
            pos_vec.reserve(new_positions.size());
        }
        pos_vec.insert(pos_vec.end(), new_positions.begin(), new_positions.end());

        // add memory for the actual integers in the position vector
        memory_delta += new_positions.size() * sizeof(uint32_t);

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

        while (i < a.postings.size() && j < b.postings.size()) {
            if (a.postings[i] < b.postings[j]) {
                result.postings.push_back(a.postings[i]);
                result.term_frequencies[a.postings[i]] = a.term_frequencies.at(a.postings[i]);
                result.positions[a.postings[i]] = a.positions.at(a.postings[i]);
                i++;
            } else if (a.postings[i] > b.postings[j]) {
                result.postings.push_back(b.postings[j]);
                result.term_frequencies[b.postings[j]] = b.term_frequencies.at(b.postings[j]);
                result.positions[b.postings[j]] = b.positions.at(b.postings[j]);
                j++;
            } else {
                // same doc_id - merge
                uint32_t doc_id = a.postings[i];
                result.postings.push_back(doc_id);
                result.term_frequencies[doc_id] = 
                    a.term_frequencies.at(doc_id) + b.term_frequencies.at(doc_id);
                
                auto pos_a = a.positions.at(doc_id);
                auto pos_b = b.positions.at(doc_id);
                pos_a.reserve(pos_a.size() + pos_b.size());
                pos_a.insert(pos_a.end(), pos_b.begin(), pos_b.end());
                result.positions[doc_id] = std::move(pos_a);
                i++;
                j++;
            }
        }
        
        while (i < a.postings.size()) {
            result.postings.push_back(a.postings[i]);
            result.term_frequencies[a.postings[i]] = a.term_frequencies.at(a.postings[i]);
            result.positions[a.postings[i]] = a.positions.at(a.postings[i]);
            i++;
        }
        
        while (j < b.postings.size()) {
            result.postings.push_back(b.postings[j]);
            result.term_frequencies[b.postings[j]] = b.term_frequencies.at(b.postings[j]);
            result.positions[b.postings[j]] = b.positions.at(b.postings[j]);
            j++;
        }
        
        return result;
    }
    
    // estimate total memory size used for dumping logic
    size_t memory_size() const {
        size_t size = postings.size() * sizeof(uint32_t);
        size += term_frequencies.size() * (MAP_NODE_OVERHEAD + sizeof(uint32_t) * 2);
        for (const auto& [doc_id, pos] : positions) {
            size += MAP_NODE_OVERHEAD + sizeof(uint32_t) + VECTOR_OVERHEAD; // map node + vec struct
            size += pos.size() * sizeof(uint32_t); // actual data
        }
        size += skip_pointers.size() * (MAP_NODE_OVERHEAD + sizeof(uint32_t) * 2);
        return size;
    }
};

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
    
    for (uint32_t doc_id : pl.postings) {
        uint32_t tf = pl.term_frequencies.at(doc_id);
        const auto& pos = pl.positions.at(doc_id);
        
        out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));
        out.write(reinterpret_cast<const char*>(&tf), sizeof(tf));
        
        uint32_t pos_count = pos.size();
        // to know how many positions to read
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

PostingList read_posting_list(std::ifstream& in, uint64_t offset) {
    PostingList pl;
    in.seekg(offset);
    
    uint32_t count_docs;
    in.read(reinterpret_cast<char*>(&count_docs), sizeof(count_docs));
    pl.postings.resize(count_docs);
    
    for (uint32_t i = 0; i < count_docs; i++) {
        uint32_t doc_id, tf, pos_count;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
        in.read(reinterpret_cast<char*>(&tf), sizeof(tf));
        in.read(reinterpret_cast<char*>(&pos_count), sizeof(pos_count));
        
        pl.postings[i] = doc_id;
        pl.term_frequencies[doc_id] = tf;
        
        std::vector<uint32_t> positions(pos_count);
        in.read(reinterpret_cast<char*>(positions.data()), pos_count * sizeof(uint32_t));
        pl.positions[doc_id] = std::move(positions);
    }
    
    return pl;
}

struct MergeState {
    std::string term;
    uint64_t offset;
    int file_index;

     // std::priority_queue is max-heap by default, so we invert the comparison
    bool operator>(const MergeState& other) const {
        return term > other.term;
    }
};

class InvertedIndexBuilder {
private:
    std::unordered_map<std::string, PostingList> partial_index;
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
            bool is_new_term = false;
            
            if (partial_index.find(term) == partial_index.end()) {
                is_new_term = true;
            }
            
            auto& pl = partial_index[term]; // existing or new PostingList
            
            size_t bytes_added = pl.add_document_occurrences(doc_id, positions);
            
            if (is_new_term) {
                // add overhead for the key in partial_index map (node + string size)
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
        
        std::vector<std::pair<const std::string*, PostingList*>> sorted_terms_ptrs;
        sorted_terms_ptrs.reserve(partial_index.size());

        for (auto& it : partial_index) {
            sorted_terms_ptrs.emplace_back(&it.first, &it.second);
        }

        std::sort(sorted_terms_ptrs.begin(), sorted_terms_ptrs.end(),
                [](auto& a, auto& b){ return *a.first < *b.first; });


        uint32_t term_count = sorted_terms_ptrs.size();
        out.write(reinterpret_cast<const char*>(&term_count), sizeof(term_count));
        
        for (auto& [term_ptr, pl_ptr] : sorted_terms_ptrs) {            
            uint32_t term_len = term_ptr->size();
            out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            out.write(term_ptr->data(), term_len);

            write_posting_list(out, *pl_ptr);
        }
        
        out.close();
        spilled_files.push_back(filename);
        
        std::cout << "Spilled " << partial_index.size() << " terms to " << filename 
                << " (" << current_memory / (1024*1024) << " MB)" << std::endl;
        
        partial_index.clear();
        current_memory = 0;
    }
    
    void finalize(const std::string& output_file_base) {
        if (!partial_index.empty()) {
            spill_to_disk();
        }

        doc_store.close();
        
        // move docstore files to final location
        if (fs::exists(output_file_base + ".docstore")) fs::remove(output_file_base + ".docstore");
        if (fs::exists(output_file_base + ".docstore_offsets")) fs::remove(output_file_base + ".docstore_offsets");
        fs::rename(temp_docstore_base + ".docstore", output_file_base + ".docstore");
        fs::rename(temp_docstore_base + ".docstore_offsets", output_file_base + ".docstore_offsets");

        std::cout << "Merging " << spilled_files.size() << " spilled files using Priority Queue..." << std::endl;
        
        std::vector<std::unique_ptr<std::ifstream>> files; 

        // queue is max-heap by default, implement min-heap by inverting comparison with MergeState::operator>
        std::priority_queue<MergeState, std::vector<MergeState>, std::greater<MergeState>> merge_queue;

        // open all temp files and push the FIRST term of each file into the queue
        // queue contains term, offset in file, file index
        // result: smallest term per file, so if e.g. "and" is smallest for 8/10 files, "or" for 2 files,
        // we have 8 entries with "and" and 2 with "or" in the queue
        for (size_t i = 0; i < spilled_files.size(); ++i) {
            const auto& filename = spilled_files[i];
            auto file = std::make_unique<std::ifstream>(filename, std::ios::binary);
            files.push_back(std::move(file));
            
            uint32_t term_count;
            // current file
            files.back()->read(reinterpret_cast<char*>(&term_count), sizeof(term_count));
            
            if (term_count > 0) {
                uint32_t term_len;
                files.back()->read(reinterpret_cast<char*>(&term_len), sizeof(term_len));
                std::string term(term_len, '\0');
                files.back()->read(&term[0], term_len);
                
                // add first term of this file to the queue
                merge_queue.push({term, (uint64_t)files.back()->tellg(), (int)i});
            }
        }
        
        // merging process
        std::ofstream postings_out(output_file_base + ".postinglists", std::ios::binary);
        std::ofstream index_out(output_file_base + ".index", std::ios::binary);
        
        uint64_t term_counter = 0;
        
        auto merge_start = std::chrono::high_resolution_clock::now();
        while (!merge_queue.empty()) {
            MergeState min_state = merge_queue.top();
            merge_queue.pop();
            std::string min_term = min_state.term;
            
            term_counter++;
            if (term_counter % 100000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - merge_start).count();
                std::cout << "Merged " << term_counter << " terms. Elapsed: " << elapsed << "s" << std::endl;
            }
            
            PostingList merged_pl;
            
            // process all entries in the queue that have the same min_term
            do { // as long as the next top term in the queue is still min_term
                int file_idx = min_state.file_index;
                
                PostingList pl = read_posting_list(*files[file_idx], min_state.offset);
                
                if (merged_pl.postings.empty()) {
                    merged_pl = std::move(pl);
                } else {
                    merged_pl = PostingList::merge(merged_pl, pl);
                }

                // read next term from this file and push to queue
                uint32_t term_len;
                if (files[file_idx]->read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) {
                    std::string next_term(term_len, '\0');
                    files[file_idx]->read(&next_term[0], term_len);
                    
                    merge_queue.push({next_term, (uint64_t)files[file_idx]->tellg(), file_idx});
                }
                
                // next term is not min_term anymore --> we cannot merge further from this file, break
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
                std::cout << "Processed " << doc_count << " documents. "
                        << "Chunk time: " << chunk_sec << " s, "
                        << "Total time: " << total_sec << " s" << std::endl;
                start_chunk = now;
            }
        }

        line_buffer = line_buffer.substr(start);
    }

    // process last line if no \n at end
    if (!line_buffer.empty()) {
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
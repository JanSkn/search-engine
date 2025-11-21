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

// TODO remove stop words?

namespace fs = std::filesystem;

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

class DocStore {
private:
    std::unordered_map<uint32_t, DocInfo> store;

public:
    void add_document(uint32_t doc_id, const std::string& url, const std::string& title) {
        store[doc_id] = {url, title};
    }

    const DocInfo* get_document(uint32_t doc_id) const {
        auto it = store.find(doc_id);
        if (it != store.end()) return &it->second;
        return nullptr;
    }

    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        uint32_t count = store.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [doc_id, info] : store) {
            uint32_t url_len = info.url.size();
            out.write(reinterpret_cast<const char*>(&url_len), sizeof(url_len));
            out.write(info.url.data(), url_len);

            uint32_t title_len = info.title.size();
            out.write(reinterpret_cast<const char*>(&title_len), sizeof(title_len));
            out.write(info.title.data(), title_len);

            out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));
        }
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        uint32_t count;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t url_len, title_len;
            in.read(reinterpret_cast<char*>(&url_len), sizeof(url_len));
            std::string url(url_len, '\0');
            in.read(&url[0], url_len);

            in.read(reinterpret_cast<char*>(&title_len), sizeof(title_len));
            std::string title(title_len, '\0');
            in.read(&title[0], title_len);

            uint32_t doc_id;
            in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
            store[doc_id] = {url, title};
        }
    }
};

// ! Snowball stemmer instance is not thread-safe!
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
    
    void add_occurrence(uint32_t doc_id, uint32_t position) {
        if (term_frequencies.find(doc_id) == term_frequencies.end()) {
            // document does not exist yet, initialize
            postings.push_back(doc_id);
            term_frequencies[doc_id] = 0;
        }
        term_frequencies[doc_id]++;
        positions[doc_id].push_back(position);
    }
    
    void build_skip_pointers() {
        size_t skip_interval = static_cast<size_t>(std::sqrt(postings.size()));

        skip_pointers.clear();
        for (size_t i = 0; i + skip_interval < postings.size(); i += skip_interval) {
            skip_pointers[i] = i + skip_interval;
        }
    }
    
    static PostingList merge(const PostingList& a, const PostingList& b) {
        PostingList result;
        size_t i = 0, j = 0;
        
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
                // Same doc_id - merge
                uint32_t doc_id = a.postings[i];
                result.postings.push_back(doc_id);
                result.term_frequencies[doc_id] = 
                    a.term_frequencies.at(doc_id) + b.term_frequencies.at(doc_id);
                
                auto pos_a = a.positions.at(doc_id);
                auto pos_b = b.positions.at(doc_id);
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
    
    // estimate memory size in bytes for memory limit for spilling
    size_t memory_size() const {
        size_t size = postings.size() * sizeof(uint32_t);
        size += term_frequencies.size() * (sizeof(uint32_t) * 2);
        for (const auto& [doc_id, pos] : positions) {
            size += pos.size() * sizeof(uint32_t);
        }
        size += skip_pointers.size() * (sizeof(uint32_t) * 2);
        return size;
    }
};

uint64_t write_posting_list(std::ofstream& out, const PostingList& pl, bool with_skip_pointers = false) {
    /*
    with_skip_pointers: we don't need skip pointers for temp spilling to disk, only for final index
    */
    uint64_t offset = out.tellp();
    uint32_t count_docs = pl.postings.size();
    // to know how many postings to read
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

class InvertedIndexBuilder {
private:
    std::unordered_map<std::string, PostingList> partial_index;
    size_t current_memory;
    size_t memory_limit;
    std::vector<std::string> spilled_files;
    int spill_counter;
    std::string temp_dir;
    DocStore doc_store;
    
public:
    InvertedIndexBuilder(size_t mem_limit_mb = 1024) 
        : current_memory(0), 
          memory_limit(mem_limit_mb * 1024 * 1024),
          spill_counter(0),
          temp_dir("temp_index") {
        fs::create_directories(temp_dir);
    }
    
    void add_document(uint32_t doc_id, const std::vector<std::string>& tokens,
                      const std::string& url = "", const std::string& title = "") {
        // temporary map to get term positions in this document
        std::unordered_map<std::string, std::vector<uint32_t>> term_positions;
        
        for (uint32_t pos = 0; pos < tokens.size(); pos++) {
            term_positions[tokens[pos]].push_back(pos);
        }
        
        for (const auto& [term, positions] : term_positions) {
            auto& pl = partial_index[term];
            size_t old_size = pl.memory_size();
            
            for (uint32_t pos : positions) {
                pl.add_occurrence(doc_id, pos);
            }
            
            size_t new_size = pl.memory_size();
            current_memory += (new_size - old_size);
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
        
        // Sort terms alphabetically for k-way merge in finalize(), WON'T WORK OTHERWISE!
        std::vector<std::pair<std::string, PostingList*>> sorted_terms;
        sorted_terms.reserve(partial_index.size());
        for (auto& [term, pl] : partial_index) {
            sorted_terms.push_back({term, &pl});
        }
        std::sort(sorted_terms.begin(), sorted_terms.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        
        uint32_t term_count = sorted_terms.size();
        out.write(reinterpret_cast<const char*>(&term_count), sizeof(term_count));
        
        // Write sorted terms
        for (auto& [term, pl_ptr] : sorted_terms) {            
            uint32_t term_len = term.size();
            out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            out.write(term.data(), term_len);
            
            write_posting_list(out, *pl_ptr);
        }
        
        out.close();
        spilled_files.push_back(filename);
        
        std::cout << "Spilled " << partial_index.size() << " terms to " << filename 
                << " (" << current_memory / (1024*1024) << " MB)" << std::endl;
        
        partial_index.clear();
        current_memory = 0;
    }
    
    void finalize(const std::string& output_file) {
        // Spill remaining data
        if (!partial_index.empty()) {
            spill_to_disk();
        }
        
        std::cout << "Merging " << spilled_files.size() << " spilled files..." << std::endl;
        
        std::vector<std::unique_ptr<std::ifstream>> files; // open all temp spilled files
        // for each file, track the following for merging
        std::vector<std::string> current_terms; // current term from each file
        std::vector<uint64_t> current_offsets; // current offset in each file
        std::vector<uint32_t> remaining_terms; // remaining terms in each file
        
        for (const auto& filename : spilled_files) {
            auto file = std::make_unique<std::ifstream>(filename, std::ios::binary);
            files.push_back(std::move(file)); // unique pointers to file streams of spilled files
            
            uint32_t term_count;
            files.back()->read(reinterpret_cast<char*>(&term_count), sizeof(term_count));
            remaining_terms.push_back(term_count);
            
            // Read first term
            if (term_count > 0) {
                uint32_t term_len;
                // read term length and reserve string of that length with null chars
                files.back()->read(reinterpret_cast<char*>(&term_len), sizeof(term_len));
                std::string term(term_len, '\0');
                files.back()->read(&term[0], term_len); // read actual term string and write into string
                current_terms.push_back(term);
                current_offsets.push_back(files.back()->tellg()); // current offset after reading term (relevant for posting list)
            } else {
                current_terms.push_back("");
                current_offsets.push_back(0);
            }
        }
        
        // Merge files
        std::ofstream postings_out(output_file + ".postinglists", std::ios::binary);
        std::ofstream index_out(output_file + ".index", std::ios::binary);
        
        while (true) {
            // Find minimum term for k-way merge -> terms are sorted in each file
            std::string min_term;
            for (const auto& term : current_terms) {
                if (!term.empty() && (min_term.empty() || term < min_term)) {
                    min_term = term;
                }
            }
            
            if (min_term.empty()) break;
            
            // Merge all posting lists for this term
            PostingList merged_pl;
            for (size_t i = 0; i < files.size(); i++) {
                if (current_terms[i] == min_term) {
                    PostingList pl = read_posting_list(*files[i], current_offsets[i]);
                    if (merged_pl.postings.empty()) {
                        merged_pl = std::move(pl);
                    } else {
                        merged_pl = PostingList::merge(merged_pl, pl);
                    }
                    
                    // Read next term from this file
                    remaining_terms[i]--;
                    if (remaining_terms[i] > 0) {
                        uint32_t term_len;
                        files[i]->read(reinterpret_cast<char*>(&term_len), sizeof(term_len));
                        std::string term(term_len, '\0');
                        files[i]->read(&term[0], term_len);
                        current_terms[i] = term;
                        current_offsets[i] = files[i]->tellg();
                    } else {
                        current_terms[i] = "";
                    }
                }
            }
            
            merged_pl.build_skip_pointers();
            uint64_t offset = write_posting_list(postings_out, merged_pl, true);
            
            // Write to index: term -> offset
            uint32_t term_len = min_term.size();
            index_out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            index_out.write(min_term.data(), term_len);
            index_out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        }
        
        postings_out.close();
        index_out.close();

        doc_store.save(output_file + ".docstore");
        
        // Cleanup temp files
        for (const auto& filename : spilled_files) {
            fs::remove(filename);
        }
        fs::remove(temp_dir);
        
        std::cout << "Index built successfully: " << output_file << std::endl;
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

// global mapping for doc IDs as strings to numeric IDs
std::unordered_map<std::string, uint32_t> docid_map;
uint32_t next_doc_id = 0;

// convert string DocID to numeric ID
uint32_t get_numeric_docid(const std::string& docid_str) {
    auto it = docid_map.find(docid_str);
    if (it != docid_map.end()) return it->second;

    uint32_t id = next_doc_id++;
    docid_map[docid_str] = id;
    return id;
}

ParsedDoc parse_line(const std::string& line) {
    size_t first_tab = line.find('\t');
    size_t second_tab = line.find('\t', first_tab + 1);
    size_t third_tab = line.find('\t', second_tab + 1);

    ParsedDoc doc;
    std::string docid_str = line.substr(0, first_tab);
    doc.doc_id = get_numeric_docid(docid_str);
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
    std::filesystem::path output_dir = project_root.parent_path() / "index" / "bin"; // put in parallel directory index where python code expects it
  
    InvertedIndexBuilder builder(std::stoull(argv[1]));

    auto start_read = std::chrono::high_resolution_clock::now();
    gzFile in = gzopen(input_file.string().c_str(), "rb");
    if (!in) {
        std::cerr << "Failed to open msmarco.tsv.gz" << std::endl;
        return 1;
    }

    char buffer[65536]; // 64 KB buffer
    uint32_t doc_count = 0;

    auto start_tokenize_add = std::chrono::high_resolution_clock::now();
    while (gzgets(in, buffer, sizeof(buffer))) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();

        auto start_parse = std::chrono::high_resolution_clock::now();
        ParsedDoc doc = parse_line(line);
        auto end_parse = std::chrono::high_resolution_clock::now();

        auto start_tokenize = std::chrono::high_resolution_clock::now();
        std::vector<std::string> tokens = tokenize(doc.body);
        auto end_tokenize = std::chrono::high_resolution_clock::now();

        auto start_add = std::chrono::high_resolution_clock::now();
        builder.add_document(doc.doc_id, tokens, doc.url, doc.title);
        auto end_add = std::chrono::high_resolution_clock::now();

        doc_count++;
        if (doc_count % 10000 == 0) {
            std::cout << "Processed " << doc_count << " documents" << std::endl;
            std::cout << "  Last parse time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - start_parse).count()
                      << " ms, tokenize: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_tokenize - start_tokenize).count()
                      << " ms, add_document: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_add - start_add).count()
                      << " ms" << std::endl;
        }
    }

    gzclose(in);
    auto end_read = std::chrono::high_resolution_clock::now();
    std::cout << "Finished reading and processing documents. Total read + process time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_read - start_read).count()
              << "s" << std::endl;

    std::filesystem::create_directories(output_dir);

    auto start_finalize = std::chrono::high_resolution_clock::now();
    builder.finalize((output_dir / "inverted_index").string());
    auto end_finalize = std::chrono::high_resolution_clock::now();

    std::cout << "Final merge time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_finalize - start_finalize).count()
              << "s" << std::endl;

    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Total indexing time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total).count()
              << "s" << std::endl;

    std::cout << "Indexing complete. Total documents: " << doc_count << std::endl;

    return 0;
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic conversion of STL containers
#include <unordered_set>
#include <unordered_map>
#include <cctype>
#include <fstream>
#include <algorithm>
#include <vector>
#include <optional>
#include "libstemmer.h"

namespace py = pybind11;

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

const std::unordered_set<std::string> KEEP_TOKENS = {"AND", "&", "OR", "|", "NOT", "-", "(", ")"};

std::vector<std::string> normalize_search_query(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;          // lowercase version for stemming
    std::string token_original; // exact original casing

    auto flush_token = [&]() {
        if (token.empty()) return;

        if (KEEP_TOKENS.find(token_original) != KEEP_TOKENS.end()) {
            tokens.push_back(token_original);          // keep original casing for operators/parentheses
        } else {
            tokens.push_back(stemmer.stem(token));
        }

        token.clear();
        token_original.clear();
    };

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            token += std::tolower(static_cast<unsigned char>(c));
            token_original += c;                       // keep original case
            continue;
        }

        // whitespace ends token
        if (std::isspace(static_cast<unsigned char>(c))) {
            flush_token();
            continue;
        }

        // special character ends token
        flush_token();

        std::string special(1, c);
        if (KEEP_TOKENS.find(special) != KEEP_TOKENS.end()) {
            tokens.push_back(special);                 // operator/punctuation
        }
    }

    // flush trailing token
    flush_token();

    return tokens;
}

struct PostingList {
    std::vector<uint32_t> postings;
    std::unordered_map<uint32_t, uint32_t> term_frequencies;
    std::unordered_map<uint32_t, std::vector<uint32_t>> positions;
    std::unordered_map<uint32_t, uint32_t> skip_pointers;

    PostingList() = default;

    PostingList(
        const std::vector<uint32_t>& p,
        const std::unordered_map<uint32_t, uint32_t>& tf,
        const std::unordered_map<uint32_t, std::vector<uint32_t>>& pos
    ) : postings(p), term_frequencies(tf), positions(pos) {}

    void build_skip_pointers() {
        size_t skip_interval = static_cast<size_t>(std::sqrt(postings.size()));

        skip_pointers.clear();
        for (size_t i = 0; i + skip_interval < postings.size(); i += skip_interval) {
            skip_pointers[i] = i + skip_interval;
        }
    }
};

PostingList read_posting_list(std::ifstream& in, uint64_t offset, bool with_skip_pointers = false) {
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
    
    if (with_skip_pointers) {
        uint32_t skip_count;
        in.read(reinterpret_cast<char*>(&skip_count), sizeof(skip_count));
        for (uint32_t i = 0; i < skip_count; i++) {
            uint32_t from_idx, to_idx;
            in.read(reinterpret_cast<char*>(&from_idx), sizeof(from_idx));
            in.read(reinterpret_cast<char*>(&to_idx), sizeof(to_idx));
            pl.skip_pointers[from_idx] = to_idx;
        }
    }
    
    return pl;
}

struct DocInfo {
    std::string url;
    std::string title;

    DocInfo() = default;

    DocInfo(const std::string& u, const std::string& t) 
        : url(u), title(t) {}
};

class DocStore {
private:
    // files for disk access
    mutable std::ifstream data_in;   
    mutable std::ifstream offset_in;
    uint32_t total_docs;

public:
    DocStore() : total_docs(0) {}

    void open(const std::string& filename_base) {
        data_in.open(filename_base + ".docstore", std::ios::binary);
        offset_in.open(filename_base + ".docstore_offsets", std::ios::binary);

        if (!data_in || !offset_in) {
            throw std::runtime_error("Could not open docstore files: " + filename_base);
        }

        // first is number of total docs
        data_in.read(reinterpret_cast<char*>(&total_docs), sizeof(total_docs));
    }

    std::optional<DocInfo> get(uint32_t doc_id) {
        if (doc_id >= total_docs) return std::nullopt;

        // offset from offset file
        uint64_t doc_offset;
        offset_in.seekg(doc_id * sizeof(uint64_t));
        if (!offset_in.read(reinterpret_cast<char*>(&doc_offset), sizeof(doc_offset))) return std::nullopt;

        data_in.seekg(doc_offset);

        uint32_t url_len;
        if (!data_in.read(reinterpret_cast<char*>(&url_len), sizeof(url_len))) return std::nullopt;
        std::string url(url_len, '\0');
        if (!data_in.read(&url[0], url_len)) return std::nullopt;

        uint32_t title_len;
        if (!data_in.read(reinterpret_cast<char*>(&title_len), sizeof(title_len))) return std::nullopt;
        std::string title(title_len, '\0');
        if (!data_in.read(&title[0], title_len)) return std::nullopt;

        return DocInfo{url, title};
    }
    
    uint32_t size() const { return total_docs; }
};

class InvertedIndex; // forward

class IndexAccessor {
private:
    InvertedIndex* parent;
public:
    IndexAccessor(InvertedIndex* p) : parent(p) {}

    std::optional<PostingList> get(const std::string& term);
};

class InvertedIndex {
private:
    std::unordered_map<std::string, uint64_t> term_to_offset;
    std::ifstream postings_file;

public:
    DocStore doc_store;
    IndexAccessor index;

    InvertedIndex(const std::string& base_path) 
        : index(this)
    {
        std::ifstream index_file(base_path + "/inverted_index.index", std::ios::binary);
        while (true) {
            uint32_t term_len;
            if (!index_file.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) break;

            std::string term(term_len, '\0');
            if (!index_file.read(&term[0], term_len)) break;

            uint64_t offset;
            if (!index_file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) break;

            term_to_offset[term] = offset;
        }

        postings_file.open(base_path + "/inverted_index.postinglists", std::ios::binary);
        if (!postings_file.is_open()) throw std::runtime_error("Cannot open postinglists");

        doc_store.open(base_path + "/inverted_index");
    }

    friend class IndexAccessor;
};

std::optional<PostingList> IndexAccessor::get(const std::string& term) {
    auto it = parent->term_to_offset.find(term);
    if (it == parent->term_to_offset.end()) return std::nullopt;
    PostingList pl = read_posting_list(parent->postings_file, it->second, true);
    return pl;
}

std::vector<uint32_t> list_union(
    const std::vector<uint32_t>& postings_1,
    const std::vector<uint32_t>& postings_2)
{
    std::vector<uint32_t> result;
    std::set_union(
        postings_1.begin(), postings_1.end(),
        postings_2.begin(), postings_2.end(),
        std::back_inserter(result)
    );
    return result;
}

std::vector<uint32_t> list_diff(
    const std::vector<uint32_t>& postings_1,
    const std::vector<uint32_t>& postings_2)
{
    std::vector<uint32_t> result;
    std::set_difference(
        postings_1.begin(), postings_1.end(),
        postings_2.begin(), postings_2.end(),
        std::back_inserter(result)
    );
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "CPP utils for search engine";

    m.def("normalize_search_query", &normalize_search_query, 
        py::arg("text"),
        "Normalize and stem search query into tokens, but keep logical operators and parentheses as is");

    m.def("list_union", &list_union,
        py::arg("postings_1"), py::arg("postings_2"),
        "Union of two sorted posting lists");

    m.def("list_diff", &list_diff,
        py::arg("postings_1"), py::arg("postings_2"),
        "Difference of two sorted posting lists (postings_1 - postings_2)");

    py::class_<DocInfo>(m, "DocInfo")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(), 
            py::arg("url"), py::arg("title")) 
        .def_readonly("url", &DocInfo::url)
        .def_readonly("title", &DocInfo::title);

    py::class_<PostingList>(m, "PostingList")
        .def(py::init<>())
        .def(py::init<
            const std::vector<uint32_t>&,
            const std::unordered_map<uint32_t, uint32_t>&,
            const std::unordered_map<uint32_t, std::vector<uint32_t>>&
        >(),
            py::arg("postings"),
            py::arg("term_frequencies"),
            py::arg("positions")
        )
        .def_readonly("postings", &PostingList::postings)
        .def_readonly("term_frequencies", &PostingList::term_frequencies)
        .def_readonly("positions", &PostingList::positions)
        .def_readonly("skip_pointers", &PostingList::skip_pointers)
        .def("build_skip_pointers", &PostingList::build_skip_pointers);

    py::class_<DocStore>(m, "DocStore")
        .def("get", &DocStore::get, py::arg("doc_id"));

    py::class_<IndexAccessor>(m, "IndexAccessor")
        .def("get", &IndexAccessor::get, py::arg("term"));

    py::class_<InvertedIndex>(m, "InvertedIndex")
        .def(py::init<const std::string&>())
        .def_readonly("index", &InvertedIndex::index)
        .def_readonly("doc_store", &InvertedIndex::doc_store);
}
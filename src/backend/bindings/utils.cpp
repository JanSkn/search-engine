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

struct Metadata {
    uint32_t num_docs = 0;
    double avg_doc_length = 0.0;
    std::unordered_map<uint32_t, uint32_t> doc_lengths;

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Cannot open metadata file");

        in.read(reinterpret_cast<char*>(&num_docs), sizeof(num_docs));
        in.read(reinterpret_cast<char*>(&avg_doc_length), sizeof(avg_doc_length));

        while (in.peek() != EOF) {
            uint32_t doc_id, length;
            if (!in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id))) break;
            if (!in.read(reinterpret_cast<char*>(&length), sizeof(length))) break;
            doc_lengths[doc_id] = length;
        }
    }

    uint32_t get_doc_length(uint32_t doc_id) const {
        auto it = doc_lengths.find(doc_id);
        if (it == doc_lengths.end()) return 0;
        return it->second;
    }
};

struct PostingList {
    std::vector<uint32_t> postings;
    uint32_t doc_frequency;
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

PostingList read_posting_list(std::ifstream& in, uint64_t offset, uint32_t docFreq) {
    PostingList pl;
    pl.doc_frequency = docFreq;
    in.seekg(offset);
    
    pl.postings.resize(docFreq);
    
    for (uint32_t i = 0; i < docFreq; i++) {
        uint32_t doc_id, pos_count;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
        in.read(reinterpret_cast<char*>(&pos_count), sizeof(pos_count));
        
        pl.postings[i] = doc_id;
        pl.term_frequencies[doc_id] = pos_count;
        
        std::vector<uint32_t> positions(pos_count);
        in.read(reinterpret_cast<char*>(positions.data()), pos_count * sizeof(uint32_t));
        pl.positions[doc_id] = std::move(positions);
    }
    
    pl.build_skip_pointers();
    
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
    std::unordered_map<uint32_t, uint64_t> offsets;
    std::ifstream data_in;
    uint32_t total_docs;

public:
    void open(const std::string& dir_name) {
        data_in.open(dir_name + "/docstore.bin", std::ios::binary);
        std::ifstream off(dir_name + "/docstore_offsets.bin", std::ios::binary);

        if (!data_in || !off)
            throw std::runtime_error("Could not open docstore");

        // docCount at the beginning
        data_in.read(reinterpret_cast<char*>(&total_docs), sizeof(total_docs));

        while (true) {
            uint32_t id;
            uint64_t off64;

            if (!off.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
            if (!off.read(reinterpret_cast<char*>(&off64), sizeof(off64))) break;

            offsets[id] = off64;
        }
    }

    std::optional<DocInfo> get(uint32_t doc_id) {
        auto it = offsets.find(doc_id);
        if (it == offsets.end()) return std::nullopt;

        uint64_t offset = it->second;
        data_in.seekg(offset);

        uint32_t url_len;
        data_in.read(reinterpret_cast<char*>(&url_len), sizeof(url_len));

        std::string url(url_len, '\0');
        data_in.read(url.data(), url_len);

        uint32_t title_len;
        data_in.read(reinterpret_cast<char*>(&title_len), sizeof(title_len));

        std::string title(title_len, '\0');
        data_in.read(title.data(), title_len);

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
    std::unordered_map<std::string, uint32_t> term_to_docfreq;
    std::ifstream postings_file;

public:
    Metadata metadata;
    DocStore doc_store;
    IndexAccessor index;

    InvertedIndex(const std::string& base_path) 
        : index(this)
    {
        std::ifstream index_file(base_path + "/index.bin", std::ios::binary);
        while (true) {
            uint32_t term_len;
            if (!index_file.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) break;

            std::string term(term_len, '\0');
            if (!index_file.read(&term[0], term_len)) break;

            uint64_t offset;
            if (!index_file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) break;

            uint32_t docFreq;
            index_file.read(reinterpret_cast<char*>(&docFreq), sizeof(docFreq));
            
            term_to_offset[term] = offset;
            term_to_docfreq[term] = docFreq;
        }

        postings_file.open(base_path + "/postinglists.bin", std::ios::binary);
        if (!postings_file.is_open()) throw std::runtime_error("Cannot open postinglists");

        metadata.load(base_path + "/metadata.bin");
        doc_store.open(base_path);
    }

    friend class IndexAccessor;
};

std::optional<PostingList> IndexAccessor::get(const std::string& term) {
    auto it = parent->term_to_offset.find(term);
    if (it == parent->term_to_offset.end()) return std::nullopt;
    uint32_t docFreq = parent->term_to_docfreq.at(term);
    PostingList pl = read_posting_list(parent->postings_file, it->second, docFreq);
    return pl;
}

PostingList positional_intersect(
    const PostingList& pl1,
    const PostingList& pl2,
    uint32_t distance
) {
    PostingList result;

    const auto& p1 = pl1.postings;
    const auto& p2 = pl2.postings;
    const auto& pos1 = pl1.positions;
    const auto& pos2 = pl2.positions;
    const auto& skip1 = pl1.skip_pointers;
    const auto& skip2 = pl2.skip_pointers;

    size_t i = 0, j = 0;
    const size_t n1 = p1.size();
    const size_t n2 = p2.size();

    result.postings.reserve(std::min(n1, n2)); // conservative

    while (i < n1 && j < n2) {
        uint32_t doc1 = p1[i];
        uint32_t doc2 = p2[j];

        if (doc1 == doc2) {
            uint32_t doc_id = doc1;

            // Check if positional data exists
            auto it1 = pos1.find(doc_id);
            auto it2 = pos2.find(doc_id);

            if (it1 != pos1.end() && it2 != pos2.end()) {
                const auto& positions1 = it1->second;
                const auto& positions2 = it2->second;

                // Positional match via two-pointer
                size_t a = 0, b = 0;
                std::vector<uint32_t> valid_positions;
                valid_positions.reserve(positions1.size());

                while (a < positions1.size() && b < positions2.size()) {
                    uint32_t pA = positions1[a];
                    uint32_t pB = positions2[b];

                    if (pB == pA + distance) {
                        valid_positions.push_back(pA);
                        ++a;
                        ++b;
                    } else if (pB < pA + distance) {
                        ++b;
                    } else {
                        ++a;
                    }
                }

                if (!valid_positions.empty()) {
                    result.postings.push_back(doc_id);
                    result.term_frequencies[doc_id] = valid_positions.size();
                    result.positions[doc_id] = std::move(valid_positions);
                }
            }

            ++i;
            ++j;
        }

        else if (doc1 < doc2) {
            // skip pointer support for pl1
            auto it_s1 = skip1.find(i);
            if (it_s1 != skip1.end() && it_s1->second < n1 && p1[it_s1->second] <= doc2) {
                i = it_s1->second;
            } else {
                ++i;
            }
        }

        else { // doc2 < doc1
            // skip pointer support for pl2
            auto it_s2 = skip2.find(j);
            if (it_s2 != skip2.end() && it_s2->second < n2 && p2[it_s2->second] <= doc1) {
                j = it_s2->second;
            } else {
                ++j;
            }
        }
    }

    // Skip pointers for result
    result.build_skip_pointers();
    return result;
}

// faster if left posting list is smaller
PostingList find_docs(
    const PostingList& pl1,
    const PostingList& pl2,
    const std::string& mode
) {
    const auto& p1 = pl1.postings;
    const auto& p2 = pl2.postings;

    const auto& skip1 = pl1.skip_pointers;
    const auto& skip2 = pl2.skip_pointers;

    const auto& tf1 = pl1.term_frequencies;
    const auto& tf2 = pl2.term_frequencies;

    size_t i = 0, j = 0;
    const size_t n1 = p1.size();
    const size_t n2 = p2.size();

    std::vector<uint32_t> result_postings;
    result_postings.reserve(std::min(n1, n2)); // most likely

    std::unordered_map<uint32_t, uint32_t> result_tf;

    if (mode == "AND") {

        while (i < n1 && j < n2) {
            uint32_t d1 = p1[i];
            uint32_t d2 = p2[j];

            if (d1 == d2) {
                result_postings.push_back(d1);
                result_tf[d1] = tf1.at(d1) + tf2.at(d2);

                i++;
                j++;
            }
            else if (d1 < d2) {
                auto it = skip1.find(i);
                if (it != skip1.end() && p1[it->second] <= d2) {
                    i = it->second;
                } else {
                    i++;
                }
            }
            else { // d2 < d1
                auto it = skip2.find(j);
                if (it != skip2.end() && p2[it->second] <= d1) {
                    j = it->second;
                } else {
                    j++;
                }
            }
        }

        PostingList out(result_postings, result_tf, {});
        out.build_skip_pointers();
        return out;
    }

    if (mode == "OR") {
        std::vector<uint32_t> merged;
        merged.reserve(n1 + n2);

        size_t a = 0, b = 0;

        while (a < n1 && b < n2) {
            uint32_t x = p1[a];
            uint32_t y = p2[b];

            if (x == y) {
                merged.push_back(x);
                result_tf[x] = tf1.at(x) + tf2.at(y);
                a++; b++;
            }
            else if (x < y) {
                merged.push_back(x);
                result_tf[x] = tf1.at(x);
                a++;
            }
            else {
                merged.push_back(y);
                result_tf[y] = tf2.at(y);
                b++;
            }
        }

        // Append remaining
        while (a < n1) {
            uint32_t x = p1[a++];
            merged.push_back(x);
            result_tf[x] = tf1.at(x);
        }

        while (b < n2) {
            uint32_t y = p2[b++];
            merged.push_back(y);
            result_tf[y] = tf2.at(y);
        }

        PostingList out(merged, result_tf, {});
        out.build_skip_pointers();
        return out;
    }

    if (mode == "NOT") {
        // result = pl1 - pl2
        std::vector<uint32_t> diff;
        diff.reserve(n1);

        size_t a = 0, b = 0;

        while (a < n1) {
            uint32_t x = p1[a];

            while (b < n2 && p2[b] < x) b++;

            if (b == n2 || p2[b] != x) {
                diff.push_back(x);
                result_tf[x] = tf1.at(x);
            }
            a++;
        }

        PostingList out(diff, result_tf, {});
        out.build_skip_pointers();
        return out;
    }

    return PostingList({}, {}, {});
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "CPP utils for search engine";

    m.def("normalize_search_query", &normalize_search_query, 
        py::arg("text"),
        "Normalize and stem search query into tokens, but keep logical operators and parentheses as is");

    m.def("positional_intersect", &positional_intersect,
        py::arg("pl1"), py::arg("pl2"), py::arg("distance") = 1,
        "Positional intersection of two posting lists with given distance");

    m.def(
        "find_docs",
        &find_docs,
        py::arg("pl1"),
        py::arg("pl2"),
        py::arg("mode"),
        "Find documents that are in both posting lists"
    );

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

    py::class_<Metadata>(m, "Metadata")
        .def_readonly("num_docs", &Metadata::num_docs)
        .def_readonly("avg_doc_length", &Metadata::avg_doc_length)
        .def_readonly("doc_lengths", &Metadata::doc_lengths)
        .def("get_doc_length", &Metadata::get_doc_length, py::arg("doc_id"));

    py::class_<DocStore>(m, "DocStore")
        .def("get", &DocStore::get, py::arg("doc_id"));

    py::class_<IndexAccessor>(m, "IndexAccessor")
        .def("get", &IndexAccessor::get, py::arg("term"));

    py::class_<InvertedIndex>(m, "InvertedIndex")
        .def(py::init<const std::string&>())
        .def_readonly("index", &InvertedIndex::index)
        .def_readonly("metadata", &InvertedIndex::metadata)
        .def_readonly("doc_store", &InvertedIndex::doc_store);
}
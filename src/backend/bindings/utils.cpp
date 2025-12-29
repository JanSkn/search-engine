#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for automatic conversion of STL containers

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libstemmer.h"

namespace py = pybind11;

bool is_valid_utf8(const std::string& s) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(s.data());
    size_t n = s.size();

    for (size_t i = 0; i < n; ++i) {
        if (b[i] <= 0x7F) continue;

        size_t len = 0;
        if ((b[i] & 0xE0) == 0xC0)
            len = 1;
        else if ((b[i] & 0xF0) == 0xE0)
            len = 2;
        else if ((b[i] & 0xF8) == 0xF0)
            len = 3;
        else
            return false;

        if (i + len >= n) return false;

        for (size_t j = 1; j <= len; ++j) {
            if ((b[i + j] & 0xC0) != 0x80) return false;
        }
        i += len;
    }
    return true;
}

std::string latin1_to_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);

    for (unsigned char c : s) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back(static_cast<char>(0xC0 | (c >> 6)));
            out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

std::string ensure_utf8(const std::string& s) {
    if (is_valid_utf8(s)) {
        return s;
    }
    return latin1_to_utf8(s);
}

struct SnowballStemmer {
    struct sb_stemmer* stemmer;
    SnowballStemmer() { stemmer = sb_stemmer_new("english", nullptr); }
    ~SnowballStemmer() { sb_stemmer_delete(stemmer); }
    std::string stem(const std::string& word) {
        const sb_symbol* stemmed =
            sb_stemmer_stem(stemmer, reinterpret_cast<const sb_symbol*>(word.c_str()), word.size());
        int out_len = sb_stemmer_length(stemmer);
        if (stemmed == nullptr || out_len <= 0) return std::string();
        return std::string(reinterpret_cast<const char*>(stemmed), static_cast<size_t>(out_len));
    }
};
SnowballStemmer stemmer;

const std::unordered_set<std::string> KEEP_TOKENS = {"AND", "&", "OR", "|", "NOT", "-", "(", ")"};

std::vector<std::string> normalize_search_query(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;           // lowercase version for stemming
    std::string token_original;  // exact original casing

    auto flush_token = [&]() {
        if (token.empty()) return;

        if (KEEP_TOKENS.find(token_original) != KEEP_TOKENS.end()) {
            tokens.push_back(token_original);  // keep original casing for operators/parentheses
        } else {
            tokens.push_back(stemmer.stem(token));
        }

        token.clear();
        token_original.clear();
    };

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            token += std::tolower(static_cast<unsigned char>(c));
            token_original += c;  // keep original case
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
            tokens.push_back(special);  // operator/punctuation
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
        std::cout << "Metadata: " << path << std::endl;
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("Cannot open metadata file");
        if (in.peek() == EOF) throw std::runtime_error("Metadata file is empty");

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

    PostingList(const std::vector<uint32_t>& p, const std::unordered_map<uint32_t, uint32_t>& tf,
                const std::unordered_map<uint32_t, std::vector<uint32_t>>& pos)
        : postings(p), term_frequencies(tf), positions(pos) {}

    void build_skip_pointers() {
        size_t skip_interval = static_cast<size_t>(std::sqrt(postings.size()));

        skip_pointers.clear();
        for (size_t i = 0; i + skip_interval < postings.size(); i += skip_interval) {
            skip_pointers[i] = i + skip_interval;
        }
    }
};

PostingList read_posting_list(std::ifstream& in, uint64_t offset, uint32_t doc_freq) {
    PostingList pl;
    pl.doc_frequency = doc_freq;
    in.seekg(offset);

    pl.postings.resize(doc_freq);

    for (uint32_t i = 0; i < doc_freq; i++) {
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

// Optimized helper to only read positions for one document for snippetting
std::vector<uint32_t> scan_posting_list_for_doc(std::ifstream& in, uint64_t offset,
                                                uint32_t doc_freq, uint32_t target_doc_id) {
    in.seekg(offset);

    for (uint32_t i = 0; i < doc_freq; i++) {
        uint32_t doc_id, pos_count;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
        in.read(reinterpret_cast<char*>(&pos_count), sizeof(pos_count));

        if (doc_id == target_doc_id) {
            std::vector<uint32_t> positions(pos_count);
            in.read(reinterpret_cast<char*>(positions.data()), pos_count * sizeof(uint32_t));
            return positions;
        }

        // if we passed the doc_id (list is sorted), it's not there
        if (doc_id > target_doc_id) {
            return {};
        }

        // skip positions for this doc
        in.seekg(pos_count * sizeof(uint32_t), std::ios::cur);
    }

    return {};
}

struct DocInfo {
    std::string url;
    std::string title;
    std::string snippet;

    DocInfo() = default;

    DocInfo(const std::string& u, const std::string& t, const std::string& s)
        : url(u), title(t), snippet(s) {}
};

class InvertedIndex;  // forward declaration

class DocStore {
   private:
    InvertedIndex* parent;
    struct DocOffset {
        uint64_t docstore_offset;  // docstore data offset
        uint64_t tsv_offset;       // offset into the msmarco tsv for body retrieval
    };
    std::unordered_map<uint32_t, DocOffset> offsets;
    std::ifstream data_in;
    std::ifstream tsv_in;
    uint32_t total_docs = 0;

    struct Hit {
        uint32_t pos;
        std::string term;
    };

    struct SubsnippetResult {
        uint32_t start;
        uint32_t end;
        std::vector<Hit> remaining_hits;
    };

    SubsnippetResult find_subsnippet(const std::vector<Hit>& hits, int max_window_size,
                                     size_t required_term_count,
                                     std::vector<std::vector<uint32_t>>& term_positions);

   public:
    std::vector<std::string> query_terms;

    DocStore(InvertedIndex* p) : parent(p) {}

    void open(const std::string& dir_name);
    std::string load_snippet(uint32_t doc_id,
                             std::vector<std::pair<uint32_t, uint32_t>>& snippet_window_borders,
                             std::vector<std::vector<uint32_t>>& term_positions,
                             uint64_t tsv_offset);
    std::string get_snippet(uint32_t doc_id, uint64_t tsv_offset);
    std::optional<DocInfo> get(uint32_t doc_id);
    std::optional<uint64_t> get_tsv_offset(uint32_t doc_id);
    uint32_t size() const { return total_docs; }
};

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

    // Query cache for performance (especially snippeting of rare + common term combos)
    std::unordered_map<std::string, std::shared_ptr<PostingList>> cache;
    void clear_cache() { cache.clear(); }

    InvertedIndex(const std::string& base_path) : doc_store(this), index(this) {
        std::ifstream index_file(base_path + "/index.bin", std::ios::binary);
        while (true) {
            uint32_t term_len;
            if (!index_file.read(reinterpret_cast<char*>(&term_len), sizeof(term_len))) break;

            std::string term(term_len, '\0');
            if (!index_file.read(&term[0], term_len)) break;

            uint64_t offset;
            if (!index_file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) break;

            uint32_t doc_freq;
            index_file.read(reinterpret_cast<char*>(&doc_freq), sizeof(doc_freq));

            term_to_offset[term] = offset;
            term_to_docfreq[term] = doc_freq;
        }

        postings_file.open(base_path + "/postinglists.bin", std::ios::binary);
        if (!postings_file.is_open()) throw std::runtime_error("Cannot open postinglists");

        metadata.load(base_path + "/metadata.bin");
        doc_store.open(base_path);
    }

    friend class DocStore;
    friend class IndexAccessor;
};

// --- Docstore ---
void DocStore::open(const std::string& dir_name) {
    data_in.open(dir_name + "/docstore.bin", std::ios::binary);
    std::string data_dir = "data";
    const char* test_env = std::getenv(
        "ENV");  // for integration tests, test with controlled and small dataset in test_data
    if (test_env && std::string(test_env) == "TEST_ENV") {
        data_dir = "test_data";
    }
    tsv_in.open(dir_name + "/../../index_builder/" + data_dir + "/msmarco-docs.tsv",
                std::ios::binary);
    std::ifstream off(dir_name + "/docstore_offsets.bin", std::ios::binary);

    if (!data_in || !tsv_in || !off) throw std::runtime_error("Could not open docstore");

    // docCount at the beginning
    data_in.read(reinterpret_cast<char*>(&total_docs), sizeof(total_docs));

    while (true) {
        uint32_t id;
        uint64_t off64;
        uint64_t tsvOff;

        if (!off.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
        if (!off.read(reinterpret_cast<char*>(&off64), sizeof(off64))) break;
        if (!off.read(reinterpret_cast<char*>(&tsvOff), sizeof(tsvOff))) break;

        offsets[id] = {off64, tsvOff};
    }
}

std::string DocStore::load_snippet(
    uint32_t doc_id, std::vector<std::pair<uint32_t, uint32_t>>& snippet_window_borders,
    std::vector<std::vector<uint32_t>>& term_positions, uint64_t tsv_offset) {
    if (snippet_window_borders.empty()) return "";

    std::sort(snippet_window_borders.begin(), snippet_window_borders.end());
    tsv_in.clear();
    tsv_in.seekg(tsv_offset);
    std::string line;
    if (!std::getline(tsv_in, line)) {
        return "";
    }

    // parse line: [DocID] \t [URL] \t [Title] \t [Content]
    // we need to find the 3rd tab to get to content
    size_t pos = 0;
    int tab_count = 0;
    while (tab_count < 3) {
        pos = line.find('\t', pos);
        if (pos == std::string::npos) return "";  // invalid format
        pos++;                                    // skip the tab
        tab_count++;
    }

    size_t content_start = pos;
    size_t len = line.size();
    std::string snippet;
    snippet.reserve(200);
    uint32_t current_word_pos = 0;
    size_t i = content_start;
    size_t window_idx = 0;

    if (snippet_window_borders[0].first > 0) {
        snippet += "... ";
    }

    // helper lambda to check if character is sentence-ending punctuation
    auto is_sentence_end = [](char c) { return c == '.' || c == '!' || c == '?'; };

    // Calculate threshold for last window (last 10%)
    uint32_t last_window_start = snippet_window_borders.back().first;
    uint32_t last_window_end = snippet_window_borders.back().second;
    uint32_t last_window_size = last_window_end - last_window_start + 1;
    uint32_t last_window_threshold = last_window_end - (last_window_size / 10);

    bool stopped_at_sentence_end = false;

    while (i < len && window_idx < snippet_window_borders.size()) {
        std::set<uint32_t> highlight_positions;
        if (window_idx < term_positions.size()) {
            highlight_positions = std::set<uint32_t>(term_positions[window_idx].begin(),
                                                     term_positions[window_idx].end());
        }
        // --- determine word ---
        size_t word_start = i;
        while (word_start < len && !std::isalnum(static_cast<unsigned char>(line[word_start]))) {
            word_start++;
        }
        std::string separator = line.substr(i, word_start - i);
        if (word_start >= len) {
            // No more words
            break;
        }
        size_t word_end = word_start;
        while (word_end < len && std::isalnum(static_cast<unsigned char>(line[word_end]))) {
            word_end++;
        }
        std::string word = line.substr(word_start, word_end - word_start);
        // ---------------------

        // check if current word is in relevant window
        // skip windows that are already passed
        while (window_idx < snippet_window_borders.size() &&
               current_word_pos > snippet_window_borders[window_idx].second) {
            window_idx++;
            if (window_idx < snippet_window_borders.size()) {
                snippet += " ... ";
            }
        }

        if (window_idx < snippet_window_borders.size()) {
            uint32_t w_start = snippet_window_borders[window_idx].first;
            uint32_t w_end = snippet_window_borders[window_idx].second;

            if (current_word_pos >= w_start && current_word_pos <= w_end) {
                // highlight a found term
                if (highlight_positions.count(current_word_pos)) {
                    word = "<b>" + word + "</b>";
                }

                if (current_word_pos == w_start) {
                    snippet += word;
                } else {
                    snippet += separator + word;
                }

                // check if we're in the last window and in its last 10%
                bool is_last_window = (window_idx == snippet_window_borders.size() - 1);
                if (is_last_window && current_word_pos >= last_window_threshold &&
                    current_word_pos < w_end) {
                    // look for sentence-ending punctuation after this word
                    size_t check_pos = word_end;
                    while (check_pos < len && check_pos < word_end + 3) {
                        if (is_sentence_end(line[check_pos])) {
                            // add it and stop early
                            snippet += line[check_pos];
                            stopped_at_sentence_end = true;
                            break;
                        }
                        check_pos++;
                    }
                    if (stopped_at_sentence_end) {
                        break;
                    }
                }
            }
        }

        // advance
        i = word_end;
        current_word_pos++;
    }

    // check if there is more text after the snippets (only if we didn't stop at sentence end)
    if (!stopped_at_sentence_end && window_idx >= snippet_window_borders.size()) {
        size_t check = i;
        while (check < len && !std::isalnum(static_cast<unsigned char>(line[check]))) check++;
        if (check < len) {
            snippet += " ...";
        }
    }

    return ensure_utf8(snippet);
}

DocStore::SubsnippetResult DocStore::find_subsnippet(
    const std::vector<Hit>& hits, int max_window_size, size_t required_term_count,
    std::vector<std::vector<uint32_t>>& term_positions) {
    SubsnippetResult result{};
    result.start = 0;
    result.end = 0;

    if (hits.empty()) return result;

    std::unordered_map<std::string, uint32_t>
        window_term_count;  // count term occurance in the window

    uint32_t left = 0;
    uint32_t best_start = hits[0].pos;
    uint32_t best_end = hits[0].pos;
    uint32_t best_score = 0;

    // mark the best window indices
    uint32_t best_left_idx = 0;
    uint32_t best_right_idx = 0;

    for (uint32_t right = 0; right < hits.size(); ++right) {
        window_term_count[hits[right].term]++;

        // shrink window if too large, adjust term counts
        while (hits[right].pos - hits[left].pos > max_window_size) {
            auto& c = window_term_count[hits[left].term];
            if (--c == 0) window_term_count.erase(hits[left].term);
            left++;
        }

        // score is number of unique terms in this window
        uint32_t score = window_term_count.size();

        // update score or choose smaller snippet -> terms more together
        if (score > best_score ||
            (score == best_score && (hits[right].pos - hits[left].pos) < (best_end - best_start))) {
            best_score = score;
            best_start = hits[left].pos;
            best_end = hits[right].pos;
            best_left_idx = left;
            best_right_idx = right;

            if (best_score == required_term_count) break;  // perfect snippet found
        }
    }

    result.start = best_start;
    result.end = best_end;

    // collect left hits (only relevant if called by first window for second window)
    result.remaining_hits.reserve(hits.size());
    for (uint32_t i = 0; i < hits.size(); ++i) {
        if (i > best_right_idx) result.remaining_hits.push_back(hits[i]);
    }

    // for highlighting positions bold
    std::vector<uint32_t> window_positions;

    for (const auto& hit : hits) {
        if (hit.pos >= best_start && hit.pos <= best_end) {
            window_positions.push_back(hit.pos);
        }
    }

    term_positions.push_back(window_positions);

    return result;
}

// total snippet length: max. MAX_WINDOW_SIZE x 2 + 1 or 2x "..."
std::string DocStore::get_snippet(uint32_t doc_id, uint64_t tsv_offset) {
    int MAX_WINDOW_SIZE = 15;  // max. num of words PER subsnippet

    if (query_terms.empty()) {
        throw std::runtime_error(
            "Set query_terms (not empty): InvertedIndex().doc_store.query_terms = ...");
    }
    std::set<std::string> unique_terms(query_terms.begin(), query_terms.end());

    // e.g.
    // hits = [ {pos: 3, term: "foo"}, {pos: 10, term: "bar"}, {pos: 15, term: "foo"}, {pos: 18,
    // term: "bar"} ]
    std::vector<Hit> hits;
    for (const auto& term : unique_terms) {
        auto cache_it = parent->cache.find(term);
        if (cache_it != parent->cache.end()) {
            const auto& pl = *cache_it->second;
            auto posIt = pl.positions.find(doc_id);
            if (posIt != pl.positions.end()) {
                for (uint32_t pos : posIt->second) hits.push_back(Hit{pos, term});
            }
            continue;
        }

        // --- term not found in cache ---
        auto termIt = parent->term_to_offset.find(term);
        if (termIt == parent->term_to_offset.end()) continue;
        auto docIt = parent->term_to_docfreq.find(term);

        std::vector<uint32_t> positions =
            scan_posting_list_for_doc(parent->postings_file, termIt->second, docIt->second, doc_id);
        if (positions.empty()) continue;

        for (uint32_t pos : positions) hits.push_back(Hit{pos, term});
        // ---------------------------------
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.pos < b.pos; });

    std::vector<std::vector<uint32_t>> term_positions;  // positions of the search terms in window i
    // example: [[1, 3, 5], [2, 5]], in window 0 (index 0): term A and B at pos. 1, 3, 5, etc.
    // create first optimal snippet
    SubsnippetResult first_snippet =
        find_subsnippet(hits, MAX_WINDOW_SIZE, unique_terms.size(), term_positions);

    // can be one if all terms fit into MAX_WINDOW_SIZE, or at most 2 for remaining terms
    // not more than 2 for readability
    std::vector<std::pair<uint32_t, uint32_t>> snippet_window_borders;
    snippet_window_borders.push_back({first_snippet.start, first_snippet.end});

    if (!first_snippet.remaining_hits.empty()) {
        SubsnippetResult second_snippet = find_subsnippet(
            first_snippet.remaining_hits, MAX_WINDOW_SIZE, unique_terms.size(), term_positions);

        if (second_snippet.end > 0) {
            snippet_window_borders.push_back({second_snippet.start, second_snippet.end});
        }
    }

    // enhance context if windows are too small
    int total_budget = MAX_WINDOW_SIZE * 2;

    if (snippet_window_borders.size() == 1) {
        // only one window --> can consume 2x the size
        auto& [start, end] = snippet_window_borders[0];
        int window_len = end - start;
        int remaining = total_budget - window_len;
        int left_context = remaining / 2;
        int right_context = remaining - left_context;

        start = (start >= left_context) ? start - left_context : 0;
        end = end + right_context;

    } else {
        // 2 windows --> half each
        int budget_per_window = total_budget / snippet_window_borders.size();

        for (auto& [start, end] : snippet_window_borders) {
            int window_len = end - start;
            int remaining = budget_per_window - window_len;
            int left_context = remaining / 2;
            int right_context = remaining - left_context;

            start = (start >= left_context) ? start - left_context : 0;
            end = end + right_context;
        }
    }

    return load_snippet(doc_id, snippet_window_borders, term_positions, tsv_offset);
}

std::optional<uint64_t> DocStore::get_tsv_offset(uint32_t doc_id) {
    auto it = offsets.find(doc_id);
    if (it == offsets.end()) return std::nullopt;
    return it->second.tsv_offset;
}

std::optional<DocInfo> DocStore::get(
    uint32_t doc_id) {  // only load snippet when required as resource-intensive
    auto it = offsets.find(doc_id);
    if (it == offsets.end()) return std::nullopt;

    uint64_t docstore_offset = it->second.docstore_offset;
    data_in.seekg(docstore_offset);
    uint64_t tsv_offset = it->second.tsv_offset;

    uint32_t url_len;
    data_in.read(reinterpret_cast<char*>(&url_len), sizeof(url_len));

    std::string url(url_len, '\0');
    data_in.read(url.data(), url_len);

    uint32_t title_len;
    data_in.read(reinterpret_cast<char*>(&title_len), sizeof(title_len));

    std::string title(title_len, '\0');
    data_in.read(title.data(), title_len);

    std::string snippet = get_snippet(doc_id, tsv_offset);
    return DocInfo{url, title, snippet};
}
// --------------------

std::optional<PostingList> IndexAccessor::get(const std::string& term) {
    // Check cache
    auto cache_it = parent->cache.find(term);
    if (cache_it != parent->cache.end()) {
        return *cache_it->second;
    }

    auto it = parent->term_to_offset.find(term);
    if (it == parent->term_to_offset.end()) return std::nullopt;
    uint32_t doc_freq = parent->term_to_docfreq.at(term);
    PostingList pl = read_posting_list(parent->postings_file, it->second, doc_freq);

    // Add to cache
    parent->cache[term] = std::make_shared<PostingList>(pl);

    return pl;
}

PostingList positional_intersect(const PostingList& pl1, const PostingList& pl2,
                                 uint32_t distance) {
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

    result.postings.reserve(std::min(n1, n2));  // conservative

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

        else {  // doc2 < doc1
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

PostingList find_docs(const PostingList& pl1, const PostingList& pl2, const std::string& mode) {
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
    result_postings.reserve(std::min(n1, n2));  // most likely

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
            } else if (d1 < d2) {
                auto it = skip1.find(i);
                if (it != skip1.end() && p1[it->second] <= d2) {
                    i = it->second;
                } else {
                    i++;
                }
            } else {  // d2 < d1
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
                a++;
                b++;
            } else if (x < y) {
                merged.push_back(x);
                result_tf[x] = tf1.at(x);
                a++;
            } else {
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

    m.def("normalize_search_query", &normalize_search_query, py::arg("text"),
          "Normalize and stem search query into tokens, but keep logical operators and parentheses "
          "as is");

    m.def("positional_intersect", &positional_intersect, py::arg("pl1"), py::arg("pl2"),
          py::arg("distance") = 1,
          "Positional intersection of two posting lists with given distance");

    m.def("find_docs", &find_docs, py::arg("pl1"), py::arg("pl2"), py::arg("mode"),
          "Find documents that are in both posting lists");

    py::class_<DocInfo>(m, "DocInfo")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&, const std::string&>(), py::arg("url"),
             py::arg("title"), py::arg("snippet"))
        .def_readonly("url", &DocInfo::url)
        .def_readonly("title", &DocInfo::title)
        .def_readonly("snippet", &DocInfo::snippet);

    py::class_<PostingList>(m, "PostingList")
        .def(py::init<>())
        .def(py::init<const std::vector<uint32_t>&, const std::unordered_map<uint32_t, uint32_t>&,
                      const std::unordered_map<uint32_t, std::vector<uint32_t>>&>(),
             py::arg("postings"), py::arg("term_frequencies"), py::arg("positions"))
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
        .def("get", &DocStore::get, py::arg("doc_id"))
        .def("get_tsv_offset", &DocStore::get_tsv_offset, py::arg("doc_id"))
        .def_readwrite("query_terms", &DocStore::query_terms);

    py::class_<IndexAccessor>(m, "IndexAccessor").def("get", &IndexAccessor::get, py::arg("term"));

    py::class_<InvertedIndex>(m, "InvertedIndex")
        .def(py::init<const std::string&>())
        .def_readonly("index", &InvertedIndex::index)
        .def_readonly("metadata", &InvertedIndex::metadata)
        .def_readonly("doc_store", &InvertedIndex::doc_store)
        .def("clear_cache", &InvertedIndex::clear_cache);
}

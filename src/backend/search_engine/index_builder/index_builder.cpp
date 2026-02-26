#include <libstemmer.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/robin_hood.h"

// not encoded as neglectably small
class DocStoreWriter {
   private:
    std::ofstream outStream;
    std::ofstream offsetStream;
    uint64_t currentByteOffset;  // offset where next doc will be written/read
    uint32_t docCount;

   public:
    void init(const std::string& filename_base) {
        outStream.open(filename_base + "/docstore.bin",
                       std::ios::binary | std::ios::out | std::ios::trunc);
        offsetStream.open(filename_base + "/docstore_offsets.bin",
                          std::ios::binary | std::ios::out | std::ios::trunc);

        currentByteOffset = 0;
        docCount = 0;

        outStream.write(reinterpret_cast<const char*>(&docCount), sizeof(docCount));
        currentByteOffset += sizeof(docCount);
    }

    void addDocument(uint32_t docId, const std::string& url, const std::string& title,
                     uint64_t tsvOffset) {
        /*
        offsetStream:
        [0-3]   docId = 42
        [4-11]  docStoreOffset = 0   (start of this doc in outStream)
        [12-19] tsvOffset = ... (start of the line of this doc in original tsv)

        [20-23] docId = 105
        [24-31] docStoreOffset = 18  (start of this doc in outStream)
        [32-39] tsvOffset = ...
        ...

        outStream:
        [0-3]   urlLen = 5
        [4-8]   'a' '.' 'c' 'o' 'm'
        [9-12]  titleLen = 5
        [13-17] 'H' 'e' 'l' 'l' 'o'

        [18-21] urlLen = 11
        [22-32] 'e' 'x' 'a' 'm' 'p' 'l' 'e' '.' 'o' 'r' 'g'
        [33-34] titleLen = 2
        [35-36] 'H' 'i'
        */
        offsetStream.write(reinterpret_cast<const char*>(&docId), sizeof(docId));
        offsetStream.write(reinterpret_cast<const char*>(&currentByteOffset),
                           sizeof(currentByteOffset));
        offsetStream.write(reinterpret_cast<const char*>(&tsvOffset), sizeof(tsvOffset));

        uint32_t urlLen = url.size();
        outStream.write(reinterpret_cast<const char*>(&urlLen), sizeof(urlLen));
        outStream.write(url.data(), urlLen);

        uint32_t titleLen = title.size();
        outStream.write(reinterpret_cast<const char*>(&titleLen), sizeof(titleLen));
        outStream.write(title.data(), titleLen);

        // faster than tellp()
        currentByteOffset += sizeof(uint32_t) + urlLen + sizeof(uint32_t) + titleLen;
        docCount++;
    }

    void close() {
        if (outStream.is_open()) {
            // write actual doc count at the beginning of the data file which was 0 before
            outStream.seekp(0);
            outStream.write(reinterpret_cast<const char*>(&docCount), sizeof(docCount));
            outStream.close();
            offsetStream.close();
        }
    }
};

struct Posting {
    int docId;
    std::vector<int> positions;
};

static const robin_hood::unordered_flat_set<std::string> STOP_WORDS = {
    "a",   "an",    "and",  "are",   "as",    "at",   "be",   "but", "by",  "for",  "if",
    "in",  "into",  "is",   "it",    "no",    "not",  "of",   "on",  "or",  "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to",  "was", "will", "with"};

class Tokenizer {
   private:
    struct sb_stemmer* stemmer;
    std::string tokenBuffer;

   public:
    Tokenizer() {
        stemmer = sb_stemmer_new("english", "UTF_8");
        if (!stemmer) {
            throw std::runtime_error("Failed to create English stemmer");
        }
        tokenBuffer.reserve(64);
    }

    ~Tokenizer() {
        if (stemmer) sb_stemmer_delete(stemmer);
    }

    // non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    template <typename Callback>
    void tokenize(const char* text, size_t len, Callback&& callback) {
        int position = 0;
        size_t i = 0;

        while (i < len) {
            while (i < len && !std::isalnum(static_cast<unsigned char>(text[i]))) {
                i++;
            }
            if (i >= len) break;

            tokenBuffer.clear();
            while (i < len && std::isalnum(static_cast<unsigned char>(text[i]))) {
                tokenBuffer.push_back(std::tolower(static_cast<unsigned char>(text[i])));
                i++;
            }

            if (tokenBuffer.empty()) continue;

            if (STOP_WORDS.count(tokenBuffer)) {
                position++;
                continue;
            }

            const sb_symbol* stemmed =
                sb_stemmer_stem(stemmer, reinterpret_cast<const sb_symbol*>(tokenBuffer.data()),
                                tokenBuffer.size());
            int stemLen = sb_stemmer_length(stemmer);

            std::string term(reinterpret_cast<const char*>(stemmed), stemLen);

            callback(std::move(term), position);
            position++;
        }
    }
};

void spillToDisk(robin_hood::unordered_flat_map<uint32_t, std::vector<Posting>>& termPostings,
                 const robin_hood::unordered_flat_map<std::string, uint32_t>& termDictionary,
                 const std::string& postingsFile, const std::string& dictFile) {
    std::vector<std::pair<std::string, uint32_t>> sortedTerms;
    sortedTerms.reserve(termDictionary.size());
    for (const auto& kv : termDictionary) sortedTerms.emplace_back(kv.first, kv.second);

    // sort for more efficient merging of the spilled files later
    std::sort(sortedTerms.begin(), sortedTerms.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::ofstream postOut(postingsFile, std::ios::binary);
    std::ofstream dictOut(dictFile, std::ios::binary);
    if (!postOut || !dictOut) throw std::runtime_error("Failed to open output files");

    static char postBuffer[8 * 1024 * 1024];  // 8MB
    static char dictBuffer[8 * 1024 * 1024];
    postOut.rdbuf()->pubsetbuf(postBuffer, sizeof(postBuffer));
    dictOut.rdbuf()->pubsetbuf(dictBuffer, sizeof(dictBuffer));

    uint64_t offset = 0;

    for (const auto& [term, termId] : sortedTerms) {
        auto it = termPostings.find(termId);
        if (it == termPostings.end()) continue;

        // sort for search and union of posting lists
        std::vector<Posting>& postings = it->second;
        std::sort(postings.begin(), postings.end(),
                  [](const Posting& a, const Posting& b) { return a.docId < b.docId; });

        uint64_t startOffset = offset;

        uint32_t docFreq = postings.size();

        for (const auto& posting : postings) {
            // write docId
            postOut.write(reinterpret_cast<const char*>(&posting.docId), sizeof(uint32_t));
            offset += sizeof(uint32_t);

            // write posCount
            uint32_t posCount = posting.positions.size();
            postOut.write(reinterpret_cast<const char*>(&posCount), sizeof(uint32_t));
            offset += sizeof(uint32_t);

            // write positions
            for (uint32_t pos : posting.positions) {
                postOut.write(reinterpret_cast<const char*>(&pos), sizeof(uint32_t));
                offset += sizeof(uint32_t);
            }
        }

        /*
        Example: apple
        [0-3]: 5 (termLen, 4 bytes)
        [4-8]: a p p l e
        [9-16]:  startOffset (8 bytes) where posting list starts in postings file
        [17-20]: docFreq (4 bytes)

        Term frequency will be in the merge step
        */
        uint32_t termLen = term.size();
        dictOut.write(reinterpret_cast<const char*>(&termLen), sizeof(termLen));
        dictOut.write(term.data(), termLen);
        dictOut.write(reinterpret_cast<const char*>(&startOffset), sizeof(startOffset));
        dictOut.write(reinterpret_cast<const char*>(&docFreq), sizeof(docFreq));
    }

    postOut.close();
    dictOut.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <memory-limit> <max-docs (optional)>" << std::endl;
        return 1;
    }
    std::cout << "Starting index building with memory limit: " << argv[1] << " MB" << std::endl;
    int32_t maxDocs = -1;
    if (argc == 3) {
        maxDocs = static_cast<int32_t>(std::stoll(argv[2]));
    }

    // high allocator and fragmentation overhead, often making real memory usage
    // 2–4× larger than the raw data size
    // --> use only 20% of the given limit for raw data
    uint64_t MaxMb = std::stoull(argv[1]);
    uint64_t MEMORYLIMIT = static_cast<uint64_t>(MaxMb * 1024ull * 1024ull * 0.20);

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    std::filesystem::path exePath = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path projectRoot = exePath.parent_path().parent_path();

    std::string dataDir = "/data";

    const char* test_env = std::getenv(
        "ENV");  // for integration tests, test with controlled and small dataset in test_data
    if (test_env && std::string(test_env) == "TEST_ENV") {
        std::cout << "TEST ENVIRONMENT, building index with test data." << std::endl;
        dataDir = "/test_data";
    }

    std::string projectDir = projectRoot.string();
    std::string partialIndexPostingsDir = projectDir + dataDir + "/partial_indices/postings";
    std::string partialIndexDictDir = projectDir + dataDir + "/partial_indices/dictionaries";
    std::string outputDir =
        (projectRoot.parent_path() / "index" / "bin")
            .string();  // put in parallel directory index/ where python code expects it
    std::string docstoreBase = outputDir;

    std::filesystem::create_directories(partialIndexPostingsDir);
    std::filesystem::create_directories(partialIndexDictDir);
    std::filesystem::create_directories(outputDir);

    Tokenizer tokenizer;
    std::ifstream infile(projectDir + dataDir + "/msmarco-docs.tsv");

    if (!infile.is_open()) {
        std::cerr << "Failed to open input file\n";
        return 1;
    }

    DocStoreWriter docStore;
    std::filesystem::create_directories(docstoreBase);
    docStore.init(docstoreBase);

    robin_hood::unordered_flat_map<std::string, uint32_t> termDictionary;
    robin_hood::unordered_flat_map<uint32_t, std::vector<Posting>> termPostings;

    // reserve space to avoid frequent rehashes, assuming 500k unique terms
    termDictionary.reserve(500'000);
    termPostings.reserve(500'000);

    std::vector<uint32_t> titleLengths;
    std::vector<uint32_t> bodyLengths;
    titleLengths.reserve(3'300'000);
    bodyLengths.reserve(3'300'000);

    std::string line;
    line.reserve(16384);
    size_t lineNumber = 0;
    uint32_t partialIndexesCount = 0;
    size_t memoryBytes = 0;

    size_t currentLineOffset =
        infile.tellg();  // store tsv file offset to restore original doc content for snippets
    while (std::getline(infile, line)) {
        lineNumber++;

        if (memoryBytes > MEMORYLIMIT) {
            std::string postingsFile = partialIndexPostingsDir + "/postings_" +
                                       std::to_string(partialIndexesCount) + ".bin";
            std::string dictFile =
                partialIndexDictDir + "/dictionary_" + std::to_string(partialIndexesCount) + ".bin";
            try {
                spillToDisk(termPostings, termDictionary, postingsFile, dictFile);
            } catch (const std::exception& e) {
                std::cerr << "Error writing index: " << e.what() << std::endl;
                return 1;
            }
            termPostings.clear();
            termDictionary.clear();
            partialIndexesCount++;
            memoryBytes = 0;
            auto elapsed = duration<double>(high_resolution_clock::now() - start).count();
            std::cout << "[Partial Index #" << partialIndexesCount << "] "
                      << "Lines processed: " << lineNumber << "  Time: " << elapsed << "s\n";
        }

        size_t pos1 = line.find('\t');
        size_t pos2 = line.find('\t', pos1 + 1);
        size_t pos3 = line.find('\t', pos2 + 1);
        if (pos3 == std::string::npos) {
            currentLineOffset = infile.tellg();
            continue;
        }

        // parse docId
        int docId = -1;
        if (pos1 >= 2 && line[0] == 'D') {
            docId = 0;
            for (size_t i = 1; i < pos1; i++) {
                char c = line[i];
                if (c >= '0' && c <= '9')
                    docId = docId * 10 + (c - '0');
                else {
                    docId = -1;
                    break;
                }
            }
        }
        if (docId < 0) {
            currentLineOffset = infile.tellg();
            continue;
        }

        std::string url = line.substr(pos1 + 1, pos2 - pos1 - 1);
        std::string title = line.substr(pos2 + 1, pos3 - pos2 - 1);

        docStore.addDocument(docId, url, title, currentLineOffset);

        // ensure length vectors are large enough
        if (static_cast<size_t>(docId) >= titleLengths.size()) {
            titleLengths.resize(docId + 1, 0);
            bodyLengths.resize(docId + 1, 0);
        }
        uint32_t titleTermCount = 0;
        uint32_t bodyTermCount = 0;

        // tokenize title + content directly
        // title is from pos2+1 to pos3, content is from pos3+1 to end
        const char* titleStart = line.data() + pos2 + 1;
        size_t titleLen = pos3 - pos2 - 1;
        const char* contentStart = line.data() + pos3 + 1;
        size_t contentLen = line.size() - pos3 - 1;

        // tokenizer reset position internally in each call, so body positions start at 0
        tokenizer.tokenize(titleStart, titleLen,
                           [&](std::string&& term, int position) { titleTermCount++; });

        // process content
        tokenizer.tokenize(contentStart, contentLen, [&](std::string&& term, int position) {
            bodyTermCount++;

            uint32_t termId;
            auto it = termDictionary.find(term);
            if (it == termDictionary.end()) {
                termId = termDictionary.size();
                memoryBytes += sizeof(uint32_t) + term.size();
                termDictionary.emplace(std::move(term), termId);
            } else {
                termId = it->second;
            }

            auto& postings = termPostings[termId];
            if (postings.empty() || postings.back().docId != docId) {
                postings.push_back({docId, {}});
                postings.back().positions.reserve(8);
                postings.back().positions.push_back(position);
                memoryBytes += sizeof(Posting) + sizeof(int);
            } else {
                postings.back().positions.push_back(position);
                memoryBytes += sizeof(int);
            }
        });

        titleLengths[docId] = titleTermCount;
        bodyLengths[docId] = bodyTermCount;

        if (maxDocs != -1 && lineNumber >= maxDocs) break;

        currentLineOffset = infile.tellg();
    }

    // final flush if remaining data
    std::string postingsFile = partialIndexPostingsDir + "/postings_final.bin";
    std::string dictFile = partialIndexDictDir + "/dictionary_final.bin";
    spillToDisk(termPostings, termDictionary, postingsFile, dictFile);

    docStore.close();

    // calculate statistics and write metadata file
    uint64_t totalTitleTerms = 0;
    uint64_t totalBodyTerms = 0;
    uint32_t numDocs = 0;
    for (size_t i = 0; i < titleLengths.size(); i++) {
        if (titleLengths[i] > 0 || bodyLengths[i] > 0) {
            totalTitleTerms += titleLengths[i];
            totalBodyTerms += bodyLengths[i];
            numDocs++;
        }
    }
    double avgTitleLength = numDocs > 0 ? static_cast<double>(totalTitleTerms) / numDocs : 0.0;
    double avgBodyLength = numDocs > 0 ? static_cast<double>(totalBodyTerms) / numDocs : 0.0;

    std::string metadataFile = outputDir + "/metadata.bin";
    std::ofstream metaOut(metadataFile, std::ios::binary);
    if (!metaOut) {
        std::cerr << "Failed to open metadata file for writing\n";
        return 1;
    }

    // write header: numDocs, avgTitleLength, avgBodyLength
    metaOut.write(reinterpret_cast<const char*>(&numDocs), sizeof(numDocs));
    metaOut.write(reinterpret_cast<const char*>(&avgTitleLength), sizeof(avgTitleLength));
    metaOut.write(reinterpret_cast<const char*>(&avgBodyLength), sizeof(avgBodyLength));

    // write document lengths array
    for (size_t docId = 0; docId < titleLengths.size(); docId++) {
        if (titleLengths[docId] > 0 || bodyLengths[docId] > 0) {
            uint32_t id = static_cast<uint32_t>(docId);
            uint32_t tLen = titleLengths[docId];
            uint32_t bLen = bodyLengths[docId];
            metaOut.write(reinterpret_cast<const char*>(&id), sizeof(id));
            metaOut.write(reinterpret_cast<const char*>(&tLen), sizeof(tLen));
            metaOut.write(reinterpret_cast<const char*>(&bLen), sizeof(bLen));
        }
    }
    metaOut.close();

    std::cout << "Metadata written: " << numDocs << " documents" << std::endl;
    std::cout << "Avg title length: " << avgTitleLength << ", Avg body length: " << avgBodyLength
              << std::endl;

    double totalTime = duration<double>(high_resolution_clock::now() - start).count();
    std::cout << "Indexing completed in " << totalTime << " seconds.\n";
    std::cout << "Total lines processed: " << lineNumber << std::endl;
    return 0;
}
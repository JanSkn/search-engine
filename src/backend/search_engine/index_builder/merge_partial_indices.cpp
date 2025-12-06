#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>
#include <queue>
#include <chrono>
#include <filesystem>
#include <map>


struct DictEntry {
    std::string term;
    uint64_t offset;
    uint32_t docFreq;
    uint64_t postingSize;  // size of the posting in bytes
};

struct PostingEntry {
    uint32_t docId;
    std::vector<uint32_t> positions;
};

std::vector<DictEntry> readDictionary(const std::string& dictFile, const std::string& postingsFile) {
    std::vector<DictEntry> entries;
    
    std::ifstream dictIn(dictFile, std::ios::binary);
    if (!dictIn) {
        std::cerr << "Failed to open dictionary file: " << dictFile << std::endl;
        return entries;
    }
    
    // get the size of the postings file to determine the last posting's size
    std::ifstream postIn(postingsFile, std::ios::binary | std::ios::ate);
    uint64_t postingsFileSize = postIn.tellg();
    postIn.close();
    
    // read all dictionary entries
    while (dictIn.peek() != EOF) {
        DictEntry entry;
        
        // read term length
        uint32_t termLen;
        dictIn.read(reinterpret_cast<char*>(&termLen), sizeof(termLen));
        if (!dictIn) break;
        
        // read term
        entry.term.resize(termLen);
        dictIn.read(&entry.term[0], termLen);
        if (!dictIn) break;
        
        // read offset
        dictIn.read(reinterpret_cast<char*>(&entry.offset), sizeof(entry.offset));
        if (!dictIn) break;
        
        // read docFreq
        dictIn.read(reinterpret_cast<char*>(&entry.docFreq), sizeof(entry.docFreq));
        if (!dictIn) break;
        
        entries.push_back(entry);
    }
    
    // calculate posting sizes using next entry's offset
    for (size_t i = 0; i < entries.size(); i++) {
        if (i + 1 < entries.size()) {
            entries[i].postingSize = entries[i + 1].offset - entries[i].offset;
        } else {
            entries[i].postingSize = postingsFileSize - entries[i].offset;
        }
    }
    
    return entries;
}

std::vector<PostingEntry> readAndParsePosting(const std::string& postingsFile, uint64_t offset, uint64_t size) {
    std::vector<PostingEntry> entries;
    
    std::ifstream postIn(postingsFile, std::ios::binary);
    if (!postIn) {
        std::cerr << "Failed to open postings file: " << postingsFile << std::endl;
        return entries;
    }
    
    postIn.seekg(offset);
    
    uint64_t bytesRead = 0;
    while (bytesRead < size) {
        PostingEntry entry;
        
        // read docId
        postIn.read(reinterpret_cast<char*>(&entry.docId), sizeof(entry.docId));
        bytesRead += sizeof(entry.docId);
        
        // read posCount
        uint32_t posCount;
        postIn.read(reinterpret_cast<char*>(&posCount), sizeof(posCount));
        bytesRead += sizeof(posCount);
        
        // read positions
        entry.positions.resize(posCount);
        postIn.read(reinterpret_cast<char*>(entry.positions.data()), posCount * sizeof(uint32_t));
        bytesRead += posCount * sizeof(uint32_t);
        
        entries.push_back(entry);
    }
    
    return entries;
}

void writePosting(std::ofstream& out, const std::vector<PostingEntry>& entries) {
    for (const auto& entry : entries) {
        // write docId
        out.write(reinterpret_cast<const char*>(&entry.docId), sizeof(entry.docId));
        
        // write posCount
        uint32_t posCount = entry.positions.size();
        out.write(reinterpret_cast<const char*>(&posCount), sizeof(posCount));
        
        // write positions
        out.write(reinterpret_cast<const char*>(entry.positions.data()), 
                  posCount * sizeof(uint32_t));
    }
}

std::vector<PostingEntry> mergePostings(const std::vector<std::vector<PostingEntry>>& allPostings) {
    // use a map to merge postings by docId (automatically sorted)
    std::map<uint32_t, std::vector<uint32_t>> mergedMap;
    
    for (const auto& postings : allPostings) {
        for (const auto& entry : postings) {
            auto& positions = mergedMap[entry.docId];
            positions.insert(positions.end(), entry.positions.begin(), entry.positions.end());
        }
    }
    
    // convert map to vector and sort positions within each document
    std::vector<PostingEntry> result;
    result.reserve(mergedMap.size());
    
    for (auto& [docId, positions] : mergedMap) {
        // sort positions for this document
        std::sort(positions.begin(), positions.end());
        
        PostingEntry entry;
        entry.docId = docId;
        entry.positions = std::move(positions);
        result.push_back(std::move(entry));
    }
    
    return result;
}

/*
Final Format:

offset file:
    term_len (4 bytes)
    term (term_len bytes)
    offset (8 bytes)
    docFreq (4 bytes)

posting file:
    docId (4 bytes)
    posCount (4 bytes)
    positions (posCount * 4 bytes)
*/
int main(int argc, char* argv[]) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    size_t termsProcessed = 0;

    std::filesystem::path exePath = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path projectRoot = exePath.parent_path().parent_path();
    std::string projectDir = projectRoot.string();

    std::string dataDir = "/data";
    
    const char* test_env = std::getenv("ENV"); // for integration tests, test with controlled and small dataset in test_data
    if (test_env && std::string(test_env) == "TEST_ENV") {
        std::cout << "TEST ENVIRONMENT, merging index with test data." << std::endl;
        dataDir = "/test_data";
    } 
    std::string partialIndexPostingsDir = projectDir + dataDir + "/partial_indices/postings";
    std::string partialIndexDictDir = projectDir + dataDir + "/partial_indices/dictionaries";
    std::string metadataDir = projectDir + dataDir + "/index";
    std::string outputDir = (projectRoot.parent_path() / "index" / "bin").string(); // put in parallel directory index/ where python code expects it
    std::filesystem::create_directories(outputDir);

    // copy from building dir to output dir
    if (std::filesystem::exists(metadataDir + "/metadata.bin")) std::filesystem::remove(outputDir + "/metadata.bin");
    std::filesystem::copy(metadataDir + "/metadata.bin", outputDir + "/metadata.bin");

    if (std::filesystem::exists(outputDir + "/docstore.bin")) std::filesystem::remove(outputDir + "/docstore.bin");
    if (std::filesystem::exists(outputDir + "/docstore_offsets.bin")) std::filesystem::remove(outputDir + "/docstore_offsets.bin");
    std::filesystem::copy(projectDir + dataDir + "/docstore/docstore.bin", outputDir + "/docstore.bin");
    std::filesystem::copy(projectDir + dataDir + "/docstore/docstore_offsets.bin", outputDir + "/docstore_offsets.bin");

    // get number of partial indices
    size_t partialIndexCount = 0;
    while (true) {
        std::string dictFile = partialIndexDictDir + "/dictionary_" + std::to_string(partialIndexCount) + ".bin";
        std::ifstream testIn(dictFile);
        if (!testIn.is_open()) break;
        testIn.close();
        partialIndexCount++;    
    }
    partialIndexCount++;
    std::cout << "Found " << partialIndexCount << " partial indices to merge.\n";

    auto allDicts = std::vector<std::vector<DictEntry>>(partialIndexCount);
    for (size_t i = 0; i < partialIndexCount - 1; i++) {
        std::string dictFile = partialIndexDictDir + "/dictionary_" + std::to_string(i) + ".bin";
        std::string postingsFile = partialIndexPostingsDir + "/postings_" + std::to_string(i) + ".bin";
        allDicts[i] = readDictionary(dictFile, postingsFile);
    }
    allDicts[partialIndexCount - 1] = readDictionary(
        partialIndexDictDir + "/dictionary_final.bin",
        partialIndexPostingsDir + "/postings_final.bin"
    );

    struct HeapEntry {
        std::string term;
        size_t dictIndex;
        size_t entryIndex;
        bool operator>(const HeapEntry& other) const {
            return term > other.term;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> minHeap;
    // initialize heap with first entry from each dictionary
    for (size_t i = 0; i < allDicts.size(); i++) {
        if (!allDicts[i].empty()) {
            minHeap.push({allDicts[i][0].term, i, 0});
        }
    }
    // open final output files
    std::string finalPostingsFile = outputDir + "/postinglists.bin";
    std::string finalDictFile = outputDir + "/index.bin";
    std::cout << "Writing merged postings to " << finalPostingsFile << std::endl;
    std::cout << "Writing merged dictionary to " << finalDictFile << std::endl;
    std::ofstream finalPostOut(finalPostingsFile, std::ios::binary);
    std::ofstream finalDictOut(finalDictFile, std::ios::binary);
    uint64_t finalOffset = 0;

    while (!minHeap.empty()) {
        auto current = minHeap.top();
        minHeap.pop();
        
        const std::string& term = current.term;
        
        // collect all postings for this term from all dictionaries
        std::vector<std::vector<PostingEntry>> postingsToMerge;
        std::vector<uint32_t> docFreqs;
        
        // add the current entry's posting
        {
            size_t dictIndex = current.dictIndex;
            size_t entryIndex = current.entryIndex;
            const DictEntry& entry = allDicts[dictIndex][entryIndex];
            docFreqs.push_back(entry.docFreq);
            
            std::string postingsFile = (dictIndex == partialIndexCount - 1)
                ? partialIndexPostingsDir + "/postings_final.bin"
                : partialIndexPostingsDir + "/postings_" + std::to_string(dictIndex) + ".bin";
            
            postingsToMerge.push_back(readAndParsePosting(postingsFile, entry.offset, entry.postingSize));
            
            if (entryIndex + 1 < allDicts[dictIndex].size()) {
                const DictEntry& nextEntry = allDicts[dictIndex][entryIndex + 1];
                minHeap.push({nextEntry.term, dictIndex, entryIndex + 1});
            }
        }
        
        // check if the next entries in the heap have the same term
        while (!minHeap.empty() && minHeap.top().term == term) {
            auto same = minHeap.top();
            minHeap.pop();
            
            size_t dictIndex = same.dictIndex;
            size_t entryIndex = same.entryIndex;
            const DictEntry& entry = allDicts[dictIndex][entryIndex];
            docFreqs.push_back(entry.docFreq);
            
            std::string postingsFile = (dictIndex == partialIndexCount - 1)
                ? partialIndexPostingsDir + "/postings_final.bin"
                : partialIndexPostingsDir + "/postings_" + std::to_string(dictIndex) + ".bin";
            
            postingsToMerge.push_back(readAndParsePosting(postingsFile, entry.offset, entry.postingSize));
            
            // push next entry from this dictionary into the heap
            if (entryIndex + 1 < allDicts[dictIndex].size()) {
                const DictEntry& nextEntry = allDicts[dictIndex][entryIndex + 1];
                minHeap.push({nextEntry.term, dictIndex, entryIndex + 1});
            }        
        }
        
        // merge postings properly (sorted by docId, with positions merged and sorted)
        std::vector<PostingEntry> mergedPostings = mergePostings(postingsToMerge);
        
        // calculate actual docFreq (number of unique documents)
        uint32_t totalDocFreq = mergedPostings.size();
        
        // write posting data to final postings file
        writePosting(finalPostOut, mergedPostings);
        
        // calculate size of merged posting
        uint64_t postingSize = 0;
        for (const auto& entry : mergedPostings) {
            postingSize += sizeof(uint32_t) + sizeof(uint32_t) + entry.positions.size() * sizeof(uint32_t);
        }
        
        // write dictionary entry to final dictionary file
        uint32_t termLen = term.size();
        finalDictOut.write(reinterpret_cast<const char*>(&termLen), sizeof(termLen));
        finalDictOut.write(term.data(), termLen);
        finalDictOut.write(reinterpret_cast<const char*>(&finalOffset), sizeof(finalOffset));
        finalDictOut.write(reinterpret_cast<const char*>(&totalDocFreq), sizeof(totalDocFreq));
        
        finalOffset += postingSize;

        termsProcessed++;

        if (termsProcessed % 100000 == 0) {
            auto now = high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            std::cout << "Processed " << termsProcessed << " terms, elapsed time: "
                    << elapsed << "s" << std::endl;
        }
    }

    std::cout << "Merging completed successfully.\n";
    std::cout << "Time taken: " 
              << duration_cast<seconds>(high_resolution_clock::now() - start).count() 
              << " seconds.\n";
    finalPostOut.close();
    finalDictOut.close();
    return 0;
}
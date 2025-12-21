import { useState, useEffect } from "react";
import { SearchInput } from "@/components/SearchInput";
import { SearchResults } from "@/components/SearchResults";
import { LoadingState } from "@/components/LoadingState";
import { ErrorState } from "@/components/ErrorState";
import { useToast } from "@/hooks/use-toast";
import { ChevronLeft, ChevronRight, Settings } from "lucide-react";
import seekrLogo from "@/assets/seekr-logo.png";

interface SearchResult {
  title: string;
  url: string;
  description?: string;
}

const Index = () => {
  const [allResults, setAllResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentQuery, setCurrentQuery] = useState("");
  const { toast } = useToast();
  const [currentText, setCurrentText] = useState("");

  // Search settings
  const [limit, setLimit] = useState(100); // Max total results
  const [tempLimit, setTempLimit] = useState(limit);
  const [resultsPerPage, setResultsPerPage] = useState(10); // Results per page
  const [showSettings, setShowSettings] = useState(false);

  const searchTexts = [
    "recipes...",
    "coding tutorials...",
    "travel destinations...",
    "workout routines...",
    "news articles...",
    "product reviews...",
  ];

  // Load page, query, limit, and results per page from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const page = parseInt(params.get("page") || "1");
    const query = params.get("q") || "";
    const urlLimit = parseInt(params.get("limit") || "100");
    const urlRPP = parseInt(params.get("rpp") || "10");

    if (page > 1) setCurrentPage(page);
    if (urlLimit) setLimit(urlLimit);
    if (urlRPP) setResultsPerPage(urlRPP);
    if (query) {
      setCurrentQuery(query);
      setHasSearched(true);
      handleSearch(query, urlLimit);
    }
  }, []);

  // Update URL whenever page, limit, or results per page changes
  useEffect(() => {
    if (hasSearched && currentQuery) {
      const params = new URLSearchParams();
      params.set("q", currentQuery);
      if (currentPage > 1) params.set("page", currentPage.toString());
      if (limit !== 10) params.set("limit", limit.toString());
      if (resultsPerPage !== 10) params.set("rpp", resultsPerPage.toString());
      window.history.pushState({}, "", `?${params.toString()}`);

      // Scroll to top when page changes
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [currentPage, limit, resultsPerPage, hasSearched, currentQuery]);

  // Typing animation effect on landing page
  useEffect(() => {
    if (hasSearched) return;

    let charIndex = 0;
    let isDeleting = false;
    let currentTextIndex = 0;
    let timeoutId: NodeJS.Timeout;

    const type = () => {
      const fullText = searchTexts[currentTextIndex];

      if (!isDeleting && charIndex <= fullText.length) {
        setCurrentText(fullText.substring(0, charIndex));
        charIndex++;
        timeoutId = setTimeout(type, 100);
      } else if (!isDeleting && charIndex > fullText.length) {
        timeoutId = setTimeout(() => {
          isDeleting = true;
          type();
        }, 2000);
      } else if (isDeleting && charIndex >= 0) {
        setCurrentText(fullText.substring(0, charIndex));
        charIndex--;
        timeoutId = setTimeout(type, 50);
      } else if (isDeleting && charIndex < 0) {
        isDeleting = false;
        charIndex = 0;
        currentTextIndex = (currentTextIndex + 1) % searchTexts.length;
        timeoutId = setTimeout(type, 500);
      }
    };

    type();

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [hasSearched]);

  // Handle search request
  const handleSearch = async (query: string, customLimit = limit) => {
    setIsLoading(true);
    setError(null);
    setHasSearched(true);
    setCurrentQuery(query);
    setCurrentPage(1);
    setAllResults([]);
    setLimit(customLimit);

    window.scrollTo({ top: 0, behavior: "smooth" });

    try {
      const response = await fetch(
        `/search?q=${encodeURIComponent(query)}&limit=${customLimit}`
      );

      if (!response.ok) {
        let errorMsg = `Search failed: ${response.status} ${response.statusText}`;
        try {
          const data = await response.json();
          if (data.detail) errorMsg = data.detail;
        } catch { }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      const searchResults = Array.isArray(data) ? data : data.results || [];
      setAllResults(searchResults);

      if (searchResults.length === 0) {
        toast({
          title: "No results found",
          description: `No results found for "${query}". Try a different search term.`,
        });
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An unknown error occurred while searching";
      console.error("Search error:", errorMessage);
      setError(errorMessage);
      setAllResults([]);
      toast({
        title: "Search failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Update max total results
  const handleLimitChange = (newLimit: number) => {
    setLimit(newLimit);
    if (hasSearched && currentQuery) {
      handleSearch(currentQuery, newLimit);
    }
    setShowSettings(false);
  };

  // Update results per page
  const handleResultsPerPageChange = (newRPP: number) => {
    setResultsPerPage(newRPP);
    setCurrentPage(1); // reset to first page
    setShowSettings(false);
  };

  // Pagination logic
  const startIndex = (currentPage - 1) * resultsPerPage;
  const endIndex = startIndex + resultsPerPage;
  const paginatedResults = allResults.slice(startIndex, endIndex);
  const totalPages = Math.ceil(allResults.length / resultsPerPage);

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setCurrentPage(newPage);
  };

  // Generate visible page numbers
  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const maxVisible = 7;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      if (currentPage <= 4) {
        for (let i = 1; i <= 5; i++) pages.push(i);
        pages.push("...");
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 3) {
        pages.push(1);
        pages.push("...");
        for (let i = totalPages - 4; i <= totalPages; i++) pages.push(i);
      } else {
        pages.push(1);
        pages.push("...");
        for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
        pages.push("...");
        pages.push(totalPages);
      }
    }

    return pages;
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Settings Button - Fixed Position */}
      {hasSearched && (
        <div className="fixed top-4 right-4 z-50">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2.5 rounded-lg bg-background border border-input hover:bg-accent transition-colors shadow-sm"
            aria-label="Settings"
          >
            <Settings className="h-5 w-5" />
          </button>

          {/* Settings Dropdown */}
          {showSettings && (
            <>
              {/* Backdrop */}
              <div
                className="fixed inset-0 z-40"
                onClick={() => setShowSettings(false)}
              />

              {/* Settings Panel */}
              <div className="absolute right-0 mt-2 w-64 bg-background border border-input rounded-lg shadow-lg p-4 z-50">
                <h3 className="font-semibold text-sm mb-3">Search Settings</h3>

                <div className="space-y-4">
                  {/* Max total results */}
                  <div className="space-y-2">
                    <label className="text-sm text-muted-foreground">Max. total results</label>
                    <input
                      type="number"
                      min={1}
                      max={500}
                      value={tempLimit}
                      onChange={(e) => setTempLimit(parseInt(e.target.value) || 1)}
                      onBlur={() => handleLimitChange(tempLimit)}          // triggers search when leaving field
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleLimitChange(tempLimit); // triggers search on Enter
                      }}
                      disabled={isLoading}
                      className="w-full px-3 py-2 text-sm border rounded-lg focus:outline-none focus:ring focus:border-primary"
                    />
                  </div>

                  {/* Results per page */}
                  <div className="space-y-2">
                    <label className="text-sm text-muted-foreground">Results per page</label>
                    <div className="grid grid-cols-2 gap-2">
                      {[5, 10, 20, 50].map((rpp) => (
                        <button
                          key={rpp}
                          onClick={() => handleResultsPerPageChange(rpp)}
                          disabled={isLoading}
                          className={`px-3 py-2 text-sm rounded-lg border transition-colors ${resultsPerPage === rpp
                              ? "bg-primary text-primary-foreground border-primary"
                              : "border-input hover:bg-accent"
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                          {rpp}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Landing Page */}
      {!hasSearched && (
        <div className="flex items-center justify-center min-h-screen px-4">
          <div className="w-full max-w-2xl">
            {/* Logo */}
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-2">
                <img src={seekrLogo} alt="Seekr logo" className="h-20 w-20" />
                <h1 className="text-7xl font-bold text-primary tracking-tight">
                  Seekr
                </h1>
              </div>
              <div className="h-6 flex items-center justify-center">
                <p className="text-muted-foreground text-lg">
                  search for <span className="text-primary font-medium">{currentText}</span>
                  <span className="animate-pulse">|</span>
                </p>
              </div>
            </div>

            {/* Search Input */}
            <SearchInput onSearch={handleSearch} isLoading={isLoading} initialValue={currentQuery} />
          </div>
        </div>
      )}

      {/* Results Page */}
      {hasSearched && (
        <div className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            {/* Compact Logo */}
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-2">
                <img src={seekrLogo} alt="Seekr logo" className="h-12 w-12" />
                <h1 className="text-4xl font-bold text-primary tracking-tight">Seekr</h1>
              </div>
            </div>

            {/* Search Input */}
            <SearchInput onSearch={handleSearch} isLoading={isLoading} initialValue={currentQuery} />

            {/* Loading State */}
            {isLoading && <LoadingState />}

            {/* Error State */}
            {error && !isLoading && <ErrorState message={error} />}

            {/* Search Results */}
            {!isLoading && paginatedResults.length > 0 && (
              <>
                <SearchResults results={paginatedResults} />

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="mt-12 mb-8">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="p-2 rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        aria-label="Previous page"
                      >
                        <ChevronLeft className="h-5 w-5" />
                      </button>

                      <div className="flex items-center gap-1">
                        {getPageNumbers().map((page, index) => (
                          <button
                            key={index}
                            onClick={() => typeof page === "number" && handlePageChange(page)}
                            disabled={page === "..."}
                            className={`min-w-[40px] h-10 rounded-lg font-medium transition-colors ${page === currentPage
                                ? "bg-primary text-primary-foreground"
                                : page === "..."
                                  ? "cursor-default"
                                  : "hover:bg-accent"
                              }`}
                          >
                            {page}
                          </button>
                        ))}
                      </div>

                      <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className="p-2 rounded-lg hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        aria-label="Next page"
                      >
                        <ChevronRight className="h-5 w-5" />
                      </button>
                    </div>

                    {/* Results info */}
                    <p className="text-center text-sm text-muted-foreground mt-4">
                      Showing {startIndex + 1}-{Math.min(endIndex, allResults.length)} of {allResults.length} results
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
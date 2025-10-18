import { Search } from "lucide-react";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchInputProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
  initialValue?: string;
}

export const SearchInput = ({ onSearch, isLoading, initialValue = "" }: SearchInputProps) => {
  const [query, setQuery] = useState(initialValue);

  // Update query when initialValue changes (e.g., from URL)
  useEffect(() => {
    setQuery(initialValue);
  }, [initialValue]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div className="flex gap-3 items-center">
        <div className="relative flex-1">
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search the web..."
            className="h-14 pl-12 pr-4 text-lg bg-card border-border shadow-md hover:shadow-lg transition-all focus:shadow-lg focus-visible:ring-2 focus-visible:ring-primary"
            disabled={isLoading}
          />
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
        </div>
        <Button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="h-14 px-8 text-base font-medium shadow-md hover:shadow-lg transition-all"
        >
          {isLoading ? "Searching..." : "Search"}
        </Button>
      </div>
    </form>
  );
};
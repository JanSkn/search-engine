import { ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/card";

interface SearchResult {
  title: string;
  url: string;
  snippet?: string;
  rrf_score?: number;
}

interface SearchResultsProps {
  results: SearchResult[];
}

export const SearchResults = ({ results }: SearchResultsProps) => {
  if (results.length === 0) {
    return null;
  }

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4 mt-8">
      {results.map((result, index) => (
        <Card
          key={index}
          className="p-5 hover:shadow-lg transition-all cursor-pointer bg-card border-border group"
        >
          <a
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <h3 className="text-xl font-semibold text-primary group-hover:underline mb-1 truncate">
                  {result.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-2 truncate">
                  {result.url}
                </p>
                {(result.snippet) && (
                  <p className="text-foreground line-clamp-2">
                    <span dangerouslySetInnerHTML={{ __html: result.snippet || "" }} />
                  </p>
                )}
                {(result.rrf_score !== undefined) && (
                  <div className="flex gap-2 mt-2">
                    {result.rrf_score !== undefined && (
                      <span className="px-2 py-1 text-xs rounded bg-blue-100 text-blue-700">
                        RRF {Number(result.rrf_score).toFixed(3)}
                      </span>
                    )}
                  </div>
                )}
              </div>
              <ExternalLink className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-1 group-hover:text-primary transition-colors" />
            </div>
          </a>
        </Card>
      ))}
    </div>
  );
};

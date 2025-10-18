import { Search } from "lucide-react";

export const EmptyState = () => {
  return (
    <div className="w-full max-w-2xl mx-auto mt-16 flex flex-col items-center gap-4 text-center">
      <div className="h-16 w-16 rounded-full bg-accent flex items-center justify-center">
        <Search className="h-8 w-8 text-accent-foreground" />
      </div>
      <p className="text-muted-foreground text-lg">
        Enter a search query to find what you're looking for
      </p>
    </div>
  );
};

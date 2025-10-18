import { Loader2 } from "lucide-react";

export const LoadingState = () => {
  return (
    <div className="w-full max-w-2xl mx-auto mt-12 flex flex-col items-center gap-4">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
      <p className="text-muted-foreground">Searching...</p>
    </div>
  );
};

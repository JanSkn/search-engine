from backend.search_engine.index.index_loader import get_index

inverted_index = get_index()
docstore = inverted_index.doc_store

print(docstore.get_title_only(3175109))


"""
uv run --project backend python -m backend.search_engine.ltr.build_data.build_dataset   --data-dir /Users/janskowron/VSCode/search-engine/src/backend/search_engine/ltr/build_data/data/msmarco   --out-dir /Users/janskowron/VSCode/search-engine/src/backend/search_engine/ltr/build_data/data/ltr_out   --index-dir /Users/janskowron/VSCode/search-engine/src/backend/search_engine/index/bin 

"""

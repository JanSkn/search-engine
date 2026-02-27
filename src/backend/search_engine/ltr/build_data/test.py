from backend.search_engine.index.index_loader import get_index

inverted_index = get_index()
docstore = inverted_index.doc_store

print(docstore.get_title_only(3175109))


'''
uv run --project backend python -m backend.search_engine.ltr.build_data.build_dataset   --data-dir /home/vietc/projects/search-engine/src/backend/search_engine/ltr/data/msmarco   --out-dir /home/vietc/projects/search-engine/src/backend/search_engine/ltr/data/ltr_out   --index-dir /home/vietc/projects/search-engine/src/backend/search_engine/index/bin 

'''
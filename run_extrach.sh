python extract_model_embedding.py -m t5-small -b 32 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m t5-base -b 8 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m t5-large -b 4 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m t5-3b -b 1 -i 128 -n 1000 -g "0"

python extract_model_embedding.py -m google/t5-v1_1-small -b 32 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/t5-v1_1-base -b 8 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/t5-v1_1-large -b 4 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/t5-v1_1-xl -b 1 -i 128 -n 1000 -g "0"

python extract_model_embedding.py -m google/flan-t5-small -b 32 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/flan-t5-base -b 8 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/flan-t5-large -b 4 -i 128 -n 1000 -g "0"
python extract_model_embedding.py -m google/flan-t5-xl -b 1 -i 128 -n 1000 -g "0"

python visualize_embedding.py -m t5-small
python visualize_embedding.py -m t5-base
python visualize_embedding.py -m t5-large
python visualize_embedding.py -m t5-3b

python visualize_embedding.py -m google/t5-v1_1-small
python visualize_embedding.py -m google/t5-v1_1-base
python visualize_embedding.py -m google/t5-v1_1-large
python visualize_embedding.py -m google/t5-v1_1-xl

python visualize_embedding.py -m google/flan-t5-small
python visualize_embedding.py -m google/flan-t5-base
python visualize_embedding.py -m google/flan-t5-large
python visualize_embedding.py -m google/flan-t5-xl
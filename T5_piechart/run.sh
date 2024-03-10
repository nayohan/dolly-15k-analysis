# # python3 generate_output_sentence.py -g 0 -m 'google/t5-v1_1-small' > 0.out &
# # python3 generate_output_sentence.py -g 1 -m 'google/flan-t5-small' > 1.out &
# # python3 generate_output_sentence.py -g 2 -m 'google/t5-v1_1-base' > 2.out &
# # python3 generate_output_sentence.py -g 3 -m 'google/flan-t5-base' > 3.out &

# # python3 generate_output_sentence_all.py -g 0 -m 'google/t5-v1_1-small' > A0.out &
# # python3 generate_output_sentence_all.py -g 1 -m 'google/flan-t5-small' > A1.out &
# # python3 generate_output_sentence_all.py -g 2 -m 'google/t5-v1_1-base' > A2.out &
# # python3 generate_output_sentence_all.py -g 3 -m 'google/flan-t5-base' > A3.out &
# # wait

# # python3 generate_output_sentence.py -g 0 -m 'google/t5-v1_1-large' > 5.out &
# # python3 generate_output_sentence.py -g 1 -m 'google/flan-t5-large' > 6.out &
# # python3 generate_output_sentence.py -g 2 -m 'google/t5-v1_1-xl' > 7.out &
# # python3 generate_output_sentence.py -g 3 -m 'google/flan-t5-xl' > 8.out &
# # wait

# # python3 generate_output_sentence_all.py -g 0 -m 'google/t5-v1_1-large' > A5.out &
# # python3 generate_output_sentence_all.py -g 1 -m 'google/flan-t5-large' > A6.out &
# # python3 generate_output_sentence_all.py -g 2 -m 'google/t5-v1_1-xl' > A7.out &
# # python3 generate_output_sentence_all.py -g 3 -m 'google/flan-t5-xl' > A8.out &
# # wait


# python3 generate_output_sentence.py -g 0 -m  't5-small' > O0.out &
# python3 generate_output_sentence.py -g 1 -m  't5-base' > O1.out &
# python3 generate_output_sentence_all.py -g 2 -m  't5-small' > O2.out &
# python3 generate_output_sentence_all.py -g 3 -m  't5-base' > O3.out &
# wait

# python3 generate_output_sentence.py -g 0 -m  't5-large' > O4.out &
# python3 generate_output_sentence.py -g 1 -m  't5-3b' > O5.out &
# python3 generate_output_sentence_all.py -g 2 -m  't5-large' > O6.out &
# python3 generate_output_sentence_all.py -g 3 -m  't5-3b' > O7.out &
# wait

# python3 generate_output_sentence_all.py -g 3 -m  'google/flan-t5-xl' > O7.out &
# python3 generate_output_sentence_all.py -g 2 -m  'google/t5-v1_1-xl' > O6.out &
python3 generate_output_sentence_all.py -g 2 -m  't5-3b' > O6.out &
# python3 boolq.py -g 0 -m 'google/t5-v1_1-small' > 1.out &
# python3 boolq.py -g 1 -m 'google/flan-t5-small' > 2.out &
# python3 boolq.py -g 2 -m 'google/t5-v1_1-base' > 3.out  &
# python3 boolq.py -g 3 -m 'google/flan-t5-base' > 4.out &
# wait
# python3 boolq.py -g 0 -m 'google/t5-v1_1-large' > 5.out &
# python3 boolq.py -g 1 -m 'google/flan-t5-large' > 6.out &
python3 boolq.py -g 2 -m 'google/t5-v1_1-xl' > 7.out &
python3 boolq.py -g 3 -m 'google/flan-t5-xl' > 8.out &
wait


# python3 piqa.py -g 0 -m 'google/t5-v1_1-small' &
# python3 piqa.py -g 1 -m 'google/flan-t5-small' &
# python3 piqa.py -g 2 -m 'google/t5-v1_1-base' &
# python3 piqa.py -g 3 -m 'google/flan-t5-base' &
# wait

# python3 siqa.py -g 0 -m 'google/t5-v1_1-small' &
# python3 siqa.py -g 1 -m 'google/flan-t5-small' &
# python3 siqa.py -g 2 -m 'google/t5-v1_1-base' &
# python3 siqa.py -g 3 -m 'google/flan-t5-base' &
# wait


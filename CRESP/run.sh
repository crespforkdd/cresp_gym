python main.py \
    --env dmc.cheetah.run \
    --agent cresp \
    -d -co -tco \
    --batch_size 256  \
    -s 0 1 2 --cuda_id 0

# python main.py \
#     --env dmc.cheetah.run \
#     --agent cresp \
#     --no-default \
#     -d -co -tco --num_sources 2 \
#     --batch_size 256  \
#     -nsr 5 -rdis 0.8 -cfc r1-k256 \
#     -s 0 1 2 --cuda_id 0

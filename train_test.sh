### Cube3
###### Train heuristic function
python train.py --env cube3 --step_max 30 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 30 --max_itrs 2000000 --num_update_procs 48 --nnet_dir models/cube3/

###### Search with trained heuristic function
python search.py --states data/cube3/test/canon.pkl --heur models/cube3/current.pt --env cube3 --batch_size 10000 --weight 1.0 --results_dir results/cube3_cannon/ --redo --time_limit 200
python search.py --states data/cube3/test/rand.pkl --heur models/cube3/current.pt --env cube3 --batch_size 10000 --weight 1.0 --results_dir results/cube3_rand/ --redo --time_limit 200

### 15-puzzle
###### Train heuristic function
python train.py --env puzzle15 --step_max 1000 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 100 --max_itrs 2000000 --num_update_procs 48 --nnet_dir models/puzzle15/


### 24-puzzle
###### Train heuristic function
python train.py --env puzzle24 --step_max 1000 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 100 --max_itrs 4000000 --num_update_procs 48 --nnet_dir models/puzzle24/


### Sokoban
###### Train heuristic function
python train.py --env sokoban --step_max 1000 --batch_size 1000 --itrs_per_update 5000 --greedy_step_update_max 30 --max_itrs 1000000 --num_update_procs 48 --nnet_dir models/sokoban/

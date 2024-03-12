### Cube3
###### Train heuristic function
python run_train.py --env cube3 --step_max 30 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 30 --max_itrs 2000000 --num_update_procs 48 --nnet_dir models/cube3/

###### Search with trained heuristic function
python run_search.py --states data/cube3/test/canon.pkl --heur models/cube3/current.pt --env cube3 --batch_size 1000 --weight 1.0 --results_dir results/cube3_cannon/ --time_limit 200
python run_search.py --states data/cube3/test/rand.pkl --heur models/cube3/current.pt --env cube3 --batch_size 1000 --weight 1.0 --results_dir results/cube3_rand/ --time_limit 200

###### Specify goals
for goal in "canon" "cross6" "cup4" "cupspot" "checkers"
do
  python run_spec_goal.py --states data/cube3/test/spec_asp.pkl --env cube3 --bk_add patterns/cube3.lp --model_batch_size 1 --heur models/cube3/current.pt --batch_size 1000 --weight 0.6 --results results_spec_asp/cube3_${goal}/ --spec "goal :- ${goal}"
done


### Sokoban
###### Train heuristic function
python run_train.py --env sokoban --step_max 1000 --batch_size 1000 --itrs_per_update 5000 --greedy_step_update_max 30 --max_itrs 1000000 --num_update_procs 48 --nnet_dir models/sokoban/

###### Search with trained heuristic function
python run_search.py --states data/sokoban/test/canon.pkl --heur models/sokoban/current.pt --env sokoban --batch_size 100 --weight 1.0 --results_dir results/sokoban/ --time_limit 200


### 15-puzzle
###### Train heuristic function
python run_train.py --env puzzle15 --step_max 1000 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 100 --max_itrs 2000000 --num_update_procs 48 --nnet_dir models/puzzle15/

###### Search with trained heuristic function
python run_search.py --states data/puzzle15/test/canon.pkl --heur models/puzzle15/current.pt --env puzzle15 --batch_size 1000 --weight 1.0 --results_dir results/puzzle15_cannon/ --time_limit 200
python run_search.py --states data/puzzle15/test/rand.pkl --heur models/puzzle15/current.pt --env puzzle15 --batch_size 1000 --weight 1.0 --results_dir results/puzzle15_rand/ --time_limit 200



### 24-puzzle
###### Train heuristic function
python run_train.py --env puzzle24 --step_max 1000 --batch_size 10000 --itrs_per_update 5000 --greedy_step_update_max 100 --max_itrs 4000000 --num_update_procs 48 --nnet_dir models/puzzle24/

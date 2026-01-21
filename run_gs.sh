pixi run python /workspace/gsplat/examples/simple_trainer_2dgs.py \
                --data_dir /workspace/glomap2k_undistorted/ \
                --data_factor 1 \
                --pose_opt \
                --app_opt \
                --normal_loss \
                --dist_loss \
                --sh_degree 3 \
                --grow_grad2d 0.00015 \
                --result_dir results/high_res_car

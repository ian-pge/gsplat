pixi run python /workspace/gsplat/examples/simple_trainer_2dgs.py \                                                                                                         2m40s160ms
          --data_dir /workspace/glomap2k_undistorted/ \
          --data_factor 1 \
          --sh_degree 3 \
          --pose_opt \
          --app_opt \
          --normal_loss \
          --dist_loss \
          --eval_steps 50000 \
          --save_steps 50000 \
          --grow_grad2d 0.0001 \
          --test_every 1000


pixi run python examples/simple_trainer.py default \
        --data-dir /workspace/undistorted \
        --result-dir /workspace/results/

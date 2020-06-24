

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/dragon_recon/dragon_vrip_pcd.ply \
    -m dragon \
    -k 50 \
    --lam 0.01


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/drill/reconstruction/drill_shaft_vrip_pcd.ply \
    -m drill \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/happy_recon/
    -m drill \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/happy_recon/happy_vrip_pcd.ply \
    -m happy \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/Armadillo_pcd.ply \
    -m armadillo \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/lucy_pcd.ply \
    -m lucy \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_dragon_pcd.ply \
    -m rgb_dragon \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_manuscript_pcd.ply \
    -m manuscript \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_statuette_pcd.ply \
    -m statuette \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/anchor.xyz \
    -m anchor \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/dc.xyz \
    -m dc \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/lord_quas.xyz \
    -m lord_quas \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/daratech.xyz \
    -m daratech \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/gargoyle.xyz \
    -m gargoyle \
    -k 50 \
    --lam 0.1

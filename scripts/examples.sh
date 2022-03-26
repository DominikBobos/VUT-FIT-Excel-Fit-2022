python3 main.py --src=../sub_eval/ --system=basedtw --feature=posteriors 
python3 main.py --src=../sub_eval/ --system=arenjansen --feature=posteriors --frame-reduction=5 
python3 main.py --src=../sub_eval/ --system=arenjansen --feature=mfcc --frame-reduction=10 
python3 main.py --src=../sub_eval/ --system=rqacluster_sdtw_unknown --feature=bottleneck --frame-reduction=20  --cluster-feature=bottleneck --parallelize-to=2000
python3 main.py --src=../sub_eval/ --system=fuzzy_match_base --feature=string 
python3 main.py --src=../sub_eval/ --system=rqacluster_fuzzy_match_unknown --feature=string --cluster-feature=bottleneck --parallelize-to=200 --parallelize-from=100 

#create clusters
python3 main.py --src=../sub_eval/ --system=rqacluster_sdtw_unknown --feature=mfcc --frame-reduction=20 --cluster-feature=mfcc
python3 main.py --src=../sub_eval/ --system=rqacluster_sdtw_unknown --feature=posteriors --frame-reduction=20 --cluster-feature=posteriors
python3 main.py --src=../sub_eval/ --system=rqacluster_sdtw_unknown --feature=bottleneck --frame-reduction=20 --cluster-feature=bottleneck

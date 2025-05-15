- currently the model only recommends items based on what the user has seen but does not take into account wether or not the user liked the content
    - want to add a like and dislike column depending on the watch ratio (thinking 0.7)     
        - in the sampler only use the liked items as the positive samples
        - in the sampler, use the disliked and unseen items as the negative samples
        - Results: change of 0.5 % from without = no change
    - want to go further and apply an upweight for the disliked items
    

    Metric comparison:
Base Metrics:
  best_val_ndcg_at_10: 0.9847658654260443
  best_val_p_at_10: 0.9847625797306889
  best_val_r_at_10: 0.9847625797306889
  corresponding_test_ndcg_at_10: 0.995293944230549
  corresponding_test_p_at_10: 0.994542877391921
  corresponding_test_r_at_10: 0.994542877391921

Explicit Negatives Metrics:
  best_val_ndcg_at_10: 0.9850541978000699
  best_val_p_at_10: 0.9841956059532259
  best_val_r_at_10: 0.9841956059532259
  corresponding_test_ndcg_at_10: 0.9918141639058667
  corresponding_test_p_at_10: 0.9883061658398312
  corresponding_test_r_at_10: 0.9883061658398312

Weighted Dislike Metrics:
  best_val_ndcg_at_10: 0.9875239632644934
  best_val_p_at_10: 0.9862508858965282
  best_val_r_at_10: 0.9862508858965282
  corresponding_test_ndcg_at_10: 0.9911129658679937
  corresponding_test_p_at_10: 0.9875974486180031
  corresponding_test_r_at_10: 0.9875974486180031
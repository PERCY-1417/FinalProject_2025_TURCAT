- currently the model only recommends items based on what the user has seen but does not take into account wether or not the user liked the content
    - want to add a like and dislike column depending on the watch ratio (thinking 0.7)     
        - in the sampler only use the liked items as the positive samples
        - in the sampler, use the disliked and unseen items as the negative samples
    - want to go further and apply an upweight for the disliked items
        
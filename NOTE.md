HR@10 

Relevance criteria is:
- 1 if the item is in the top 10 recommendations
- 0 otherwise

Process we make a items_to_rank of 10 true_positive items (coming from test set) and 100 random items (coming from all items)

We then use the model to score the items in items_to_rank

We then sort the items in items_to_rank by the model scores

We then take the top 10 items and calculate the HR@10




We decided to go with SASRec because it’s great at handling sequential data, which is exactly what we need for recommending short videos. Since users on platforms like Kuaishou typically watch videos in a sequence, SASRec can capture patterns in how people interact with content over time. Unlike basic methods that just look at overall user-item relationships, SASRec uses a Transformer model to focus on the order in which videos are watched, helping us predict what a user might want to watch next. It's a solid model for this kind of task, and it’s been shown to work really well for similar recommendation problems. Plus, it's efficient and scalable, making it a good fit for our project’s goals.

These articles were very informative and gave a broader understanding of SASRec

https://medium.com/biased-algorithms/contrastive-learning-for-sequential-recommendation-f4744d75128a

https://medium.com/@rohan.chaudhury.rc/paper-review-self-attentive-sequential-recommendation-a4efd2185a61
# Trust_Recommendation
Trust Based Recommendation systems

Read the following in order:

1. TrustAwareRecommenderSystems_Report.pdf
2. Trust Aware Recommendation Systems.pdf
3. BTP Report 2.pdf
4. BTP Evaluation 3.pdf
5. BTP Final Report.pdf	

I basically implemented multiple papers on trust recommendations:

- ste.py
- sorec.py
- socialmf.py

You'll find the PDF of each of the papers in here as well.

Them I made optimised version of them, which take waaay less time to run:

- este.py
- esorec.py

("e" here stands for efficiency)

I made a hybrid approach which "I think" gave better results than both, don't remember exactly.
hybrid_sorec.py

Finally there exists only one trust and item rating data set currently which is epinion, I created another one personally by scraping the site https://www.zomato.com/ . This data set must be in the Zomato folder. As far as I remember I made it in the same structure as the Epinions dataset.
The difference between this and epinions is that this has way more trust relations for a group of users. 
Here is a comparision between epinions and zomato:

| Parameters\Datasets | Epinions | Zomato |
| --- | --- | --- |
| Users | 49,290 | 8028 |
| Items | 1,39,738 | 92,324 |
| Item Ratings | 6,64,823 | 2,33,974 |
| Trust Relations | 4,87,182 | 38,84,886 |

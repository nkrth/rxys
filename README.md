# rxys

Tired of stochastic gradient descent in SVD-like algorithms, we turn to more interesting ideas like Hybrid systems, Alternating Least Squares (ALS) and Implicit Feedback. Perhaps, ARIMA and LSTM ensembled with Fast R-CNN in a large stacknet. In this project, however, we use a concept from Swarm Intelligence known as PSO to optimize suggestions given to us by what once was a multi-context text generator.

This project is related to https://github.com/NicolasHug/Surprise and has implementations like Amazon Product Recommendations, Facebook Friend Suggestions and Book Suggestions as in https://github.com/dorukkilitcioglu/books2rec

Please read the docs for more on the available implementations. (docs will be made available shortly)

**Note: Sorry, I've taken out the implementation files from the push queue. Trying to make them as easy to access as possible.**

You can generate your own instance of an optimized recommender using: 

` 
!pip install rxys
import rxys
rxys.init()
rxys.sample()
`

This works on Python 3.7.x and backward compatibility is not supported.
If this doesn't work, our PyPI is broken. It will be up shortly.

For any suggestions, feedback or clarifications, reach me through e-mail at nick.kartha@gmail.com (cc: kartha@vivaldi.net for faster response)

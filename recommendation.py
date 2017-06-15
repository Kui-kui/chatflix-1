# coding: utf-8

from User import User
from random import randint

import numpy as np
import operator

from sklearn.cluster import KMeans

from movielens import load_movies, load_simplified_ratings


class Recommendation:

    def __init__(self):

        # Importe la liste des films
        # Dans la variable 'movies' se trouve la correspondance entre l'identifiant d'un film et le film
        # Dans la variables 'movies_list' se trouve les films populaires qui sont vus par les utilisateurs
        self.movies = load_movies()
        self.movies_list = []

        # Importe la liste des notations
        # Dans le tableau 'ratings' se trouve un objet avec un attribut 'movie' contenant l'identifiant du film, un
        # attribut 'user' avec l'identifiant de l'utilisateur et un attribut 'is_appreciated' pour savoir si oui ou non
        # l'utilisateur aime le film
        self.ratings = load_simplified_ratings()

        # Les utilisateurs du fichier 'ratings-popular-simplified.csv' sont stockés dans 'test_users'
        self.test_users = {}
        # Les utilisateurs du chatbot facebook seront stockés dans 'users'
        self.users = {}

        # Lance le traitement des notations
        self.process_ratings_to_users()

    # Traite les notations
    # Crée un utilisateur de test pour chaque utilisateur dans le fichier
    # Puis lui attribue ses films aimés et détestés
    def process_ratings_to_users(self):
        for rating in self.ratings:
            user = self.register_test_user(rating.user)
            if rating.is_appreciated is not None:
                if rating.is_appreciated:
                    user.good_ratings.append(rating.movie)
                else:
                    user.bad_ratings.append(rating.movie)
            elif rating.score is not None:
                user.ratings.append(rating)
            self.movies_list.append(rating.movie)

    # Enregistre un utilisateur de test s'il n'existe pas déjà et le retourne
    def register_test_user(self, sender):
        if sender not in self.test_users.keys():
            self.test_users[sender] = User(sender)
        return self.test_users[sender]

    # Enregistre un utilisateur s'il n'existe pas déjà et le retourne
    def register_user(self, sender):
        if sender not in self.users.keys():
            self.users[sender] = User(sender)
        return self.users[sender]

    # Retourne les films aimés par un utilisateur
    def get_movies_from_user(self, user):
        movies_list = []
        good_movies = user.good_ratings
        for movie_number in good_movies:
            movies_list.append(self.movies[movie_number].title)
        return movies_list

    # Affiche la recommandation pour l'utilisateur
    def make_recommendation(self, user):
        similarities = self.compute_all_similarities(user)

        test_user_id = max(similarities.items(), key=operator.itemgetter(1))[0]
        test_user = self.test_users[test_user_id]

        movies = dict([(movie.id, movie.title) for movie in self.movies])
        return str([movies[movie_id] for movie_id in test_user.good_ratings])

    # Pose une question à l'utilisateur
    def ask_question(self, user):
        i = randint(0, len(self.movies_list) - 1)
        movie = self.movies[i]
        user.set_question(movie.id)

        return "Hello Nikui, tu as aimé {} ?".format(movie.title)

    # Calcule la similarité entre 2 utilisateurs
    @staticmethod
    def get_similarity(user_a, user_b):
        ans = 0
        good_ratings_b = set(user_b.good_ratings)
        bad_ratings_b = set(user_b.bad_ratings)
        neutral_ratings_b = set(user_b.neutral_ratings)

        for good_rating in user_a.good_ratings:
            if good_rating in good_ratings_b:
                ans += 2
            elif good_rating in bad_ratings_b:
                ans -= 2

        for bad_rating in user_a.bad_ratings:
            if bad_rating in bad_ratings_b:
                ans += 2
            elif bad_rating in good_ratings_b:
                ans -= 2

        for neutral_rating in user_a.neutral_ratings:
            if neutral_rating in neutral_ratings_b:
                ans += 1

        return ans / (user_a.get_norm() + user_b.get_norm())

    # Calcule la similarité entre un utilisateur et tous les utilisateurs de tests
    def compute_all_similarities(self, user):
        similarities = {}
        for other_user in self.test_users.values():
            similarities[other_user.id] = Recommendation.get_similarity(user, other_user)

        return similarities

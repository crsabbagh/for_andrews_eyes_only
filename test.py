import django
django.setup()

import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, mode
from update_data.models import Player, Team, Boxscore
from django.db.models import Sum, Count, FloatField
from django.db.models.functions import Cast


# Query the boxscore database: [{'player__id_number': int, 'player__name': str, 'stat': float}, ...]
stats = Boxscore.objects.all()\
    .filter(is_playoffs=0)\
    .values('player__id_number',
            'player__name',
            'blocks',
            'minutes_played')\
    .order_by('player')

# Initialize variables
player_dict = {}
total = 0

print('Reformatting data')

for s in stats:

    # Add player to the dictionary if he has not yet been seen
    if s['player__id_number'] not in player_dict:
        player_dict[s['player__id_number']] = {'name': s['player__name'],
                                               'stats': [],
                                               'weight': 0,
                                               'distribution': [],
                                               'minutes': 0}

    # Append statistic to the stats list
    player_dict[s['player__id_number']]['stats'].append(s['blocks'])

    # Add minutes for future filtering
    player_dict[s['player__id_number']]['minutes'] += s['minutes_played']

    # Add total number of the state in the season
    total += s['blocks']

print('Initial pool contains {0} players'.format(len(player_dict)))

# Initialize variables
minutes_threshold = 500

print('Fitting distributions')

for p in list(player_dict.keys()):

    # Initialize weight parameter to stat_p / sum(stat_p for all p)
    # player_dict[p]['weight'] = sum(player_dict[p]['stats']) / total

    # If player played fewer than the minutes threshold, delete them
    if player_dict[p]['minutes'] < minutes_threshold:
        del player_dict[p]

    # Otherwise, fit a skewed normal distribution to the player's stats
    else:
        player_dict[p]['distribution'] = skewnorm.fit(player_dict[p]['stats'])

print('Filtered pool contains {0} players'.format(len(player_dict)))
print('Making predictions')

# Initialize variables
debug = False
iterations = 1000001
print_iterations = (iterations - 1) / 100
epsilon = 1
player_ids = list(player_dict.keys())
time1 = time.time()
num_correct = 0

for i in range(iterations):

    # Randomly generate teams
    random.shuffle(player_ids)
    prediction = [0, 0]
    reality = [0, 0]

    # Iterate through both teams
    for j in range(len(prediction)):

        start = 0 if j == 0 else 10
        end = 10 if j == 0 else 20

        # Iterate through the first team (first 10 players)
        for team1_player in player_ids[start:end]:

            # Add weight to the prediction
            prediction[j] += player_dict[team1_player]['weight']

            # Determine how many games each player will play this week based on actual season probabilities
            num = random.randint(1, 100)
            if num <= 15:
                num_games = 2
            elif num <= 62:
                num_games = 3
            elif num <= 99:
                num_games = 4
            else:
                num_games = 5

            # Sample the statistic distribution per the number of games and add it to the total
            contributions = np.round(skewnorm(player_dict[team1_player]['distribution'][0],
                                              player_dict[team1_player]['distribution'][1],
                                              player_dict[team1_player]['distribution'][2])
                                     .rvs(num_games))

            reality[j] += np.sum(contributions)

            if debug:
                print('Team {0}: {1} plays {2} games and contributes {3}'.format(
                    j + 1,
                    player_dict[team1_player]['name'],
                    num_games,
                    contributions
                ))

    if debug:
        print('\nPrediction:\n'
              '\tTeam 1: {0}\n'
              '\tTeam 2: {1}\n\n'
              'Actual:\n'
              '\tTeam1: {2}\n'
              '\tTeam2: {3}'.format(
                prediction[0],
                prediction[1],
                reality[0],
                reality[1]
                ))

    # If the prediction is correct, leave the weights as is
    if (prediction[0] > prediction[1] and reality[0] > reality[1]) or \
        (prediction[1] > prediction[0] and reality[1] > reality[0]):
        num_correct += 1
        if debug:
            print('Prediction was correct, no adjustments made')

    # Otherwise, increase winner weights by epsilon and decrease loser weights by epsilon
    elif reality[0] > reality[1]:
        for player_id in player_ids[:10]:
            player_dict[player_id]['weight'] += epsilon
        for player_id in player_ids[11:20]:
            player_dict[player_id]['weight'] -= epsilon

        if debug:
            print('Prediction was wrong, team 1 is getting a boost')

    elif reality[1] < reality[0]:
        for player_id in player_ids[:10]:
            player_dict[player_id]['weight'] -= epsilon
        for player_id in player_ids[11:20]:
            player_dict[player_id]['weight'] += epsilon

        if debug:
            print('Prediction was wrong, team 2 is getting a boost')

    # Every 1000 passes, print the leaders in the statistics and their weights
    if i % print_iterations == 0 or debug:

        time2 = time1
        time1 = time.time()

        print('\n\tIteration number {0}\n'
              '\t{1} seconds since last post\n'
              '\t{2}% of predictions were correct'.format(i, time1 - time2, (num_correct / print_iterations) * 100))
        sorted_dict = sorted(player_dict.items(), key=lambda x: x[1]['weight'], reverse=True)

        for j in sorted_dict[:10]:
            print('\t\t{0}: {1}'.format(j[1]['name'], j[1]['weight']))

        num_correct = 0

    if debug and i == 3:
        break

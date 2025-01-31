import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import random
import time
import os

from scipy.stats import entropy
from scipy.special import expit
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib 
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.8)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.8)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.8)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=12)
    parser.add_argument('--option_temperature', help="Temperature parameter for softmax", type=float, default=1e-1)
    parser.add_argument('--action_temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--seed', help="seed", type=int, default=1)
    
    parser.add_argument('--multi_option', help="multi updates", default=False, action='store_true')
    parser.add_argument('--eta', help="Lambda for EOC", type=float, default=1.0)
    parser.add_argument('--linear_eta', help="Multi-updates state-dependent linear hyperparameter", default=False, action='store_true')
    parser.add_argument('--linear_eta_distance', help='Distance concept for Linear eta [eucliedean, cityblock]', type=str, default='euclidean')
    
    args = parser.parse_args()
    # Multi-updates state-dependent linear hyperparameter - based on distance concept
    if args.linear_eta:
        eta_state_dependent = StateDependentETA(args.linear_eta_distance)
        print("State dependent eta!")
    elif args.eta:
        eta=args.eta
        print(f"Eta fixed at {eta}")
    args.lr_term =args.lr_intra
    total_steps=0
    start=time.time()
    possible_next_goals = [74,75,84,85]

    history_steps = np.zeros((args.nruns, args.nepisodes))
    observation_history = np.zeros((args.nruns, args.nepisodes, args.noptions, 13, 13)) # 13x13 room
    state_eta_history = np.zeros((args.nruns, args.nepisodes, 13, 13))
    
    for run in range(args.nruns):
        env = gym.make('Fourrooms-v0')
        env.set_goal(62)
        env.set_seed(args.seed+run)


        np.random.seed(args.seed+run)
        random.seed(args.seed+run)

        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n


        # The intra-option policies are linear-softmax functions
        option_policies = [FixedPolicy(nfeatures, nactions, o) for o in range(args.noptions)]


        # The termination function are linear-sigmoid functions
        option_terminations = FixedTermination()

        # The initiation set
        initiation_set = FixedInitiationSet()

        # Policy over options
        _policy = SoftmaxPolicy(nfeatures, args.noptions, args.option_temperature)

        meta_policy = InitiationSetSoftmaxPolicy(args.noptions,initiation_set, _policy)

        # Critic over options
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, _policy.weights, _policy, args.noptions, meta_policy) 



        tot_steps=0.
        for episode in range(args.nepisodes):


            
            last_opt=None
            phi = features(env.reset())
            option = meta_policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)



            action_ratios_avg=[]
            
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                next_phi = features(observation)

                if option_terminations.sample(next_phi,option): 
                    next_option = meta_policy.sample(next_phi)
                else:
                    next_option=option

                next_action = option_policies[next_option].sample(next_phi)



                ###Action ratios
                action_ratios=np.zeros((args.noptions))
                for o in range(args.noptions):
                    action_ratios[o] = option_policies[o].pmf(phi)[action]
                action_ratios= action_ratios / action_ratios[option]
                action_ratios_avg.append(action_ratios)


                # Prob of current option
                one_hot = np.zeros(args.noptions)
                if last_opt is not None:
                    bet = np.float(option_terminations.pmf(phi,last_opt))
                    one_hot[last_opt] = 1.
                else:
                    bet = 1.0
                prob_curr_opt = bet * np.array(meta_policy.pmf(phi)) + (1-bet)*one_hot
                one_hot_curr_opt= np.zeros(args.noptions)
                one_hot_curr_opt[option] = 1.
                
                # print("State: ", phi, type(phi), env.tocell[phi.item()])
                if args.linear_eta:
                    current_state = env.tocell[phi.item()]
                    current_i, current_j = current_state
                    current_goal = env.tocell[env.goal]
                    eta_given_phi = eta_state_dependent.eta_wrt_goal(current_state, current_goal)
                    prob_curr_opt= eta_given_phi * prob_curr_opt + (1-eta_given_phi) * one_hot_curr_opt
                    #tracking cell activity in room
                    state_eta_history[run, episode, current_i, current_j] = eta_given_phi

                elif args.eta:    
                    sampled_eta = float(np.random.rand() < eta)
                    prob_curr_opt= eta * prob_curr_opt + (1-eta) * one_hot_curr_opt
                    current_state = env.tocell[phi.item()]
                    current_i, current_j = current_state
                    #tracking cell activity in room
                    state_eta_history[run, episode, current_i, current_j] = eta
                
                else:
                    pass

                # Critic updates
                if args.multi_option:
                    critic.update(next_phi, next_option, reward, done, prob_curr_opt * action_ratios)
                else:
                    critic.update(next_phi, next_option, reward, done, one_hot_curr_opt)

                #tracking cell activity in room
                i,j = env.tocell[phi.item()]
                observation_history[run, episode, option, i, j] += 1

                last_opt=option
                phi=next_phi
                option=next_option
                action=next_action


                if done:
                    break


            tot_steps+=step
            history_steps[run, episode] = step
            end=time.time()
            print('Run {} Total steps {} episode {} steps {} FPS {:0.0f} '.format(run,tot_steps, episode, step,   int(tot_steps/ (end- start)) )  )

    if args.multi_option:
        new_folder_name = f'Fixedop_multi_option_noptions{args.noptions}_nruns{args.nruns}_nepisodes{args.nepisodes}_nsteps{args.nsteps}'
    else:
        new_folder_name = f'Fixedop_single_option_noptions{args.noptions}_nruns{args.nruns}_nepisodes{args.nepisodes}_nsteps{args.nsteps}'
    if args.linear_eta:
        new_folder_name += f"_LINEAR_eta_distance_{args.linear_eta_distance}"
    else:
        new_folder_name += f"_fixed_eta_{args.eta}"

    if os.path.exists(new_folder_name):
        pass 
    else:
        os.mkdir(new_folder_name)
        print("new folder created: ", new_folder_name)

    #saving obsrvations and history
    import pickle as pkl 
    with open(os.path.join(new_folder_name, "history_steps.pkl"), "wb") as f:
        pkl.dump(history_steps, f)
    with open(os.path.join(new_folder_name,"observation_history.pkl"), "wb") as f:
        pkl.dump(observation_history, f)
    with open(os.path.join(new_folder_name,"state_eta_history.pkl"), "wb") as f:
        pkl.dump(state_eta_history, f)

    #plottng grahs
    avg_steps_per_episode = exponential_smoothing(np.mean(history_steps, axis=0))
    CI = 0.95
    plt.plot(range(args.nepisodes), avg_steps_per_episode, color='blue', label='Avg steps in completion with ' + str(CI) + '% CI')
    #confidence interval
    ci = CI * np.std(avg_steps_per_episode)/np.mean(avg_steps_per_episode)
    plt.fill_between(range(args.nepisodes), (avg_steps_per_episode-ci), (avg_steps_per_episode+ci), color='cyan', alpha=0.3)
    plt.xlabel("Episodes")
    plt.ylabel("Avg steps")
    plt.savefig(os.path.join(new_folder_name, "Graph1.jpg"))
    plt.close("all")
    
    activity_per_run = np.mean(observation_history, axis=0)
    activity_per_episode = np.mean(activity_per_run, axis=0)
    
    #colormap for visited states wrt. options
    for option in range(args.noptions):
        print('Option no.: ', option)
        full_env = activity_per_episode[option,:,:]
        new_virdis = cm.get_cmap('viridis', 5)
        plt.pcolormesh(full_env, cmap = new_virdis)
        plt.colorbar()
        plt.title(f"Colormap of freq. of visited states in Option {option}")
        plt.savefig(os.path.join(new_folder_name, "Colormap_option"+str(option)+'.jpg'))
        plt.close("all")

    #colormap for state_related eta
    eta_state_activity_per_run = np.mean(state_eta_history, axis=0)
    eta_state_activity_per_episode = np.sum(eta_state_activity_per_run, axis=0)
    eta_full_env = eta_state_activity_per_episode
    new_inferno = cm.get_cmap('inferno', 5)
    plt.pcolormesh(eta_full_env, cmap = new_inferno)
    plt.colorbar()
    plt.title('Colormap for eta in visited states')
    plt.savefig(os.path.join(new_folder_name, "Colormap_linear_eta_state.jpg"))
    plt.close("all")

    print("Completed!")
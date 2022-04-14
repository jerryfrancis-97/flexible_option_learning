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
import seaborn as sns
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
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--option_temperature', help="Temperature parameter for softmax", type=float, default=1e-1)
    parser.add_argument('--action_temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--seed', help="seed", type=int, default=1)


    parser.add_argument('--multi_option', help="multi updates", default=False, action='store_true')
    parser.add_argument('--eta', help="Multi-updates hypereparameter", type=float, default=0.3)
    parser.add_argument('--linear_eta', help="Multi-updates state-dependent linear hyperparameter", default=False, action='store_true')
    parser.add_argument('--linear_eta_distance', help='Distance concept for Linear eta [eucliedean, cityblock]', type=str, default='euclidean')
    parser.add_argument('--new_randomness', help="new degree of randomness", type=float, default=0.45)

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
    termination_option_next_phi_history = np.zeros((args.nruns, args.nepisodes, args.noptions, 13, 13))
    option_update_entropy_history = np.zeros((args.nruns, args.nepisodes, args.noptions, 13, 13))
    kl_history = np.zeros((args.nruns, args.nepisodes, args.noptions, 13, 13))
    for run in range(args.nruns):
        env = gym.make('Fourrooms-v0')
        env.set_goal(62)
        env.set_seed(args.seed+run)


        np.random.seed(args.seed+run)
        random.seed(args.seed+run)

        features = Tabular(env.observation_space.n)
        nfeatures, nactions = len(features), env.action_space.n


        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(nfeatures, nactions, args.action_temperature) for _ in range(args.noptions)]

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(nfeatures) for _ in range(args.noptions)]

        # Policy over options
        meta_policy = SoftmaxPolicy(nfeatures, args.noptions, args.option_temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, meta_policy.weights, meta_policy,args.noptions) 

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term,args.noptions)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra, args.discount, critic,args.noptions)
        


        tot_steps=0.
        for episode in range(args.nepisodes):
            if episode > 0 and episode == int(args.nepisodes/2.): ############################# Change time #############################
                goal=possible_next_goals[args.seed % len(possible_next_goals)]
                env.set_goal(goal)
                env.set_randomness(args.new_randomness)
                print('************* New goal : ', env.goal)



            
            last_opt=None
            phi = features(env.reset())
            option = meta_policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)



            action_ratios_avg=[]
            
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                next_phi = features(observation)

                if option_terminations[option].sample(next_phi): 
                    next_option = meta_policy.sample(next_phi)
                else:
                    next_option=option

                next_action = option_policies[next_option].sample(next_phi)

                #tracking termination for current option in the next state
                #HCECK if current option terminates in next state
                is_terminated_option_next_phi = int(option_terminations[option].sample(next_phi))
                next_phi_i, next_phi_j = env.tocell[next_phi.item()] 
                termination_option_next_phi_history[run, episode, option, next_phi_i, next_phi_j] = is_terminated_option_next_phi

                ###Action ratios
                action_ratios=np.zeros((args.noptions))
                for o in range(args.noptions):
                    action_ratios[o] = option_policies[o].pmf(phi)[action]
                action_ratios= action_ratios / action_ratios[option]
                action_ratios_avg.append(action_ratios)


                # Prob of current option
                one_hot = np.zeros(args.noptions)
                if last_opt is not None:
                    bet = option_terminations[last_opt].pmf(phi)
                    one_hot[last_opt] = 1.
                else:
                    bet = 1.0
                prob_curr_opt = bet * meta_policy.pmf(phi) + (1-bet)*one_hot
                prob_termination_effect_option = prob_curr_opt
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
                    # tracking normal entropy of multi-update in current option
                    option_update_entropy_history[run, episode, option, current_i, current_j] = Entropy(prob_curr_opt[option])
                    # tracking kl-divergence of multi-update prob in current option when prob. of termination of option is known
                    kl_history[run, episode, option, current_i, current_j] = KL_Divergence(prob_curr_opt[option], prob_termination_effect_option[option])
                

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
                critic.update(next_phi, next_option, reward, done, one_hot_curr_opt)


                # Intra-option policy update
                critic_feedback = reward + args.discount * critic.value(next_phi, next_option)
                critic_feedback -= critic.value(phi, option)
                if args.multi_option:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        action_ratios, prob_curr_opt  )   
                else:
                    intraoption_improvement.update(phi, option, action, reward, done, next_phi, next_option, critic_feedback,
                        np.ones_like(action_ratios), one_hot_curr_opt  ) 

                # Termination update
                if not done:
                    termination_improvement.update(next_phi, option, one_hot_curr_opt )

                #tracking cell activity in room
                i,j = env.tocell[phi.item()]
                observation_history[run, episode, option, i, j] = 1


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

    new_folder_name = f'output/' # comment this for running in colab
    time_of_creation = time.ctime()
    if args.multi_option:
        new_folder_name += f'Exp@{time_of_creation}_Learned_multi_option_noptions{args.noptions}_nruns{args.nruns}_nepisodes{args.nepisodes}_nsteps{args.nsteps}'
    else:
        new_folder_name += f'Exp@{time_of_creation}_Learned_single_option_noptions{args.noptions}_nruns{args.nruns}_nepisodes{args.nepisodes}_nsteps{args.nsteps}'
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
    with open(os.path.join(new_folder_name,"termination_option_next_phi_history.pkl"), "wb") as f:
        pkl.dump(termination_option_next_phi_history, f)
    with open(os.path.join(new_folder_name,"option_update_entropy_history.pkl"), "wb") as f:
        pkl.dump(option_update_entropy_history, f)
    with open(os.path.join(new_folder_name,"kl_history.pkl"), "wb") as f:
        pkl.dump(kl_history, f)


    #plottng grahs
    avg_steps_per_episode = exponential_smoothing(np.mean(history_steps, axis=0), weight=0.6)
    CI = 0.95
    plt.plot(range(args.nepisodes), avg_steps_per_episode, color='blue', label='Avg steps in completion with ' + str(CI) + '% CI')
    #confidence interval
    ci = CI * np.std(avg_steps_per_episode)/np.mean(avg_steps_per_episode)
    plt.fill_between(range(args.nepisodes), (avg_steps_per_episode-ci), (avg_steps_per_episode+ci), color='cyan', alpha=0.3)
    plt.xlabel("Episodes")
    plt.ylabel("Avg steps")
    plt.savefig(os.path.join(new_folder_name, "Graph1.jpg"))
    plt.close("all")
    
    
    #colormap for visited states wrt. options
    activity_per_run = np.mean(observation_history, axis=0)
    activity_per_episode = np.mean(activity_per_run, axis=0)
    for option in range(args.noptions):
        print('Option no.: ', option)
        full_env = activity_per_episode[option,:,:]
        new_virdis = cm.get_cmap('viridis', 5)
        plt.pcolormesh(full_env, cmap = new_virdis)
        plt.colorbar()
        plt.title(f"Frequency of visited states in Option {option}")
        plt.savefig(os.path.join(new_folder_name, "Colormap_option"+str(option)+'.jpg'))
        plt.close("all")

    #colormap for termination states wrt. options
    termination_per_run = np.mean(termination_option_next_phi_history, axis=0)
    termination_per_episode = np.mean(termination_per_run, axis=0)
    for option in range(args.noptions):
        print('Option no.: ', option)
        full_env = termination_per_episode[option,:,:]
        new_cividis = cm.get_cmap('cividis', 5)
        plt.pcolormesh(full_env, cmap = new_cividis)
        plt.colorbar()
        plt.title(f"Termination states in Option {option}")
        plt.savefig(os.path.join(new_folder_name, "Colormap_Termination_option"+str(option)+'.jpg'))
        plt.close("all")

    #colormap for entrpoy of option updates wrt. options
    option_update_per_run = np.mean(option_update_entropy_history, axis=0)
    option_update_per_episode = np.mean(option_update_per_run, axis=0)
    for option in range(args.noptions):
        print('Option no.: ', option)
        full_env = option_update_per_episode[option,:,:]
        new_cividis = cm.get_cmap('plasma', 5)
        plt.pcolormesh(full_env, cmap = new_cividis)
        plt.colorbar()
        plt.title(f"Entropy of Option {option}")
        plt.savefig(os.path.join(new_folder_name, "Colormap_Entropy_option"+str(option)+'.jpg'))
        plt.close("all")

    #colormap for KL of option updates wrt. prob. of termination of current options
    kl_per_run = np.mean(kl_history, axis=0)
    kl_per_episode = np.mean(kl_per_run, axis=0)
    for option in range(args.noptions):
        print('Option no.: ', option)
        full_env = kl_per_episode[option,:,:]
        new_cividis = cm.get_cmap('plasma', 5)
        plt.pcolormesh(full_env, cmap = new_cividis)
        plt.colorbar()
        plt.title(f"KL of Option {option} wrt. to Termination")
        plt.savefig(os.path.join(new_folder_name, "Colormap_KL_option"+str(option)+'.jpg'))
        plt.close("all")

    #colormap for state_related eta
    eta_state_activity_per_run = np.mean(state_eta_history, axis=0)
    eta_state_activity_per_episode = np.sum(eta_state_activity_per_run, axis=0)
    eta_full_env = eta_state_activity_per_episode
    new_inferno = cm.get_cmap('inferno', 5)
    plt.pcolormesh(eta_full_env, cmap = new_inferno)
    plt.colorbar()
    plt.title('Eta in visited states')
    plt.savefig(os.path.join(new_folder_name, "Colormap_linear_eta_state.jpg"))
    plt.close("all")

    print("Completed!")
import logging
from typing import Any, List, Union
import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except (ModuleNotFoundError, ImportError) as e:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

from citylearn.agents.rbc import RBC
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from citylearn.preprocessing import Encoder, RemoveFeature
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork

class SAC(RLC):
    ###Modify SAC.__init__ to Load Constraint Critic 06.02.2025
    def __init__(self, env: CityLearnEnv, constraint_model, **kwargs: Any):
        r"""Custom soft actor-critic algorithm.

        Parameters
        ----------
        env: CityLearnEnv
            CityLearn environment.
        
        Other Parameters
        ----------------
        **kwargs : Any
            Other keyword arguments used to initialize super class.
        """

        super().__init__(env, **kwargs)

        ###Modify SAC.__init__ to Load Constraint Critic 06.02.2025
        self.constraint_model = constraint_model  # Load pre-trained constraint model
        self.lagrange_multiplier = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Lagrange λ
        self.lagrange_optimizer = optim.Adam([self.lagrange_multiplier], lr=0.01)

        # internally defined
        self.normalized = [False for _ in self.action_space]
        self.soft_q_criterion = nn.SmoothL1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = [ReplayBuffer(int(self.replay_buffer_capacity)) for _ in self.action_space]
        self.soft_q_net1 = [None for _ in self.action_space]
        self.soft_q_net2 = [None for _ in self.action_space]
        self.target_soft_q_net1 = [None for _ in self.action_space]
        self.target_soft_q_net2 = [None for _ in self.action_space]
        self.policy_net = [None for _ in self.action_space]
        self.soft_q_optimizer1 = [None for _ in self.action_space]
        self.soft_q_optimizer2 = [None for _ in self.action_space]
        self.policy_optimizer = [None for _ in self.action_space]
        self.target_entropy = [None for _ in self.action_space]
        self.norm_mean = [None for _ in self.action_space]
        self.norm_std = [None for _ in self.action_space]
        self.r_norm_mean = [None for _ in self.action_space]
        self.r_norm_std = [None for _ in self.action_space]
        self.set_networks()

    
#Modify SAC.update() to Include Constraint Penalty 0.6.02.2025 /// Hmmm to be discussed !!!
    
    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float], next_observations: List[List[float]], terminated: bool, truncated: bool):
        
        
        r"""Update replay buffer.

        Parameters
        ----------
        observations : List[List[float]]
            Previous time step observations.
        actions : List[List[float]]
            Previous time step actions.
        reward : List[float]
            Current time step reward.
        next_observations : List[List[float]]
            Current time step observations.
        terminated : bool
            Indication that episode has ended.
        truncated : bool
            If episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).

        for i, (o, a, r, n) in enumerate(zip(observations, actions, reward, next_observations)):
            o = self.get_encoded_observations(i, o)
            n = self.get_encoded_observations(i, n)

            if self.normalized[i]:
                o = self.get_normalized_observations(i, o)
                n = self.get_normalized_observations(i, n)
                r = self.get_normalized_reward(i, r)
            else:
                pass
            # Get constraint value from pre-trained model
         ###   with torch.no_grad():
           ###     constraint_value = self.constraint_model(torch.tensor(np.hstack([o, a]), dtype=torch.float32)).item()
            # Modify reward function to include constraint penalty
          ###  modified_reward = r - self.lagrange_multiplier.item() * constraint_value

            # Store updated transition
          ###  self.replay_buffer[i].push(o, a, modified_reward, n, terminated)
        
            self.replay_buffer[i].push(o, a, r, n, terminated)

            if self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i]):
                if not self.normalized[i]:
                    # calculate normalized observations and rewards
                    X = np.array([j[0] for j in self.replay_buffer[i].buffer], dtype = float)
                    self.norm_mean[i] = np.nanmean(X, axis=0)
                    self.norm_std[i] = np.nanstd(X, axis=0) + 1e-5
                    R = np.array([j[2] for j in self.replay_buffer[i].buffer], dtype = float)
                    self.r_norm_mean[i] = np.nanmean(R, dtype = float)
                    self.r_norm_std[i] = np.nanstd(R, dtype = float)/self.reward_scaling + 1e-5
                    
                    # update buffer with normalization
                    self.replay_buffer[i].buffer = [(
                        np.hstack(self.get_normalized_observations(i, o).reshape(1,-1)[0]),
                        a,
                        self.get_normalized_reward(i, r),
                        np.hstack(self.get_normalized_observations(i, n).reshape(1,-1)[0]),
                        d
                    ) for o, a, r, n, d in self.replay_buffer[i].buffer]
                    self.normalized[i] = True
                
                else:
                    pass

                for _ in range(self.update_per_time_step):
                    o, a, r, n, d = self.replay_buffer[i].sample(self.batch_size)
                    tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                    o = tensor(o).to(self.device)
                    n = tensor(n).to(self.device)
                    a = tensor(a).to(self.device)
                    r = tensor(r).unsqueeze(1).to(self.device)
                    d = tensor(d).unsqueeze(1).to(self.device)

                    with torch.no_grad():
                        # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) observation and its associated log probability of occurrence.
                        new_next_actions, new_log_pi, _ = self.policy_net[i].sample(n)

                        # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                        target_q_values = torch.min(
                            self.target_soft_q_net1[i](n, new_next_actions),
                            self.target_soft_q_net2[i](n, new_next_actions),
                        ) - self.alpha*new_log_pi
                        q_target = r + (1 - d)*self.discount*target_q_values

                    # Update Soft Q-Networks
                    q1_pred = self.soft_q_net1[i](o, a)
                    q2_pred = self.soft_q_net2[i](o, a)
                    q1_loss = self.soft_q_criterion(q1_pred, q_target)
                    q2_loss = self.soft_q_criterion(q2_pred, q_target)
                    self.soft_q_optimizer1[i].zero_grad()
                    q1_loss.backward()
                    self.soft_q_optimizer1[i].step()
                    self.soft_q_optimizer2[i].zero_grad()
                    q2_loss.backward()
                    self.soft_q_optimizer2[i].step()

                    # Update Policy
                    new_actions, log_pi, _ = self.policy_net[i].sample(o)

                    #added 06.02.2025
                    # Query constraint value for current state-action pair
                    constraint_value = self.constraint_model(torch.cat([o, new_actions], dim=-1)).detach()

                    
                    q_new_actions = torch.min(
                        self.soft_q_net1[i](o, new_actions),
                        self.soft_q_net2[i](o, new_actions)
                    )
                    
                    #added 06.02.2025
                    # Modify policy loss to include constraint penalty
                    policy_loss = (-q_new_actions + self.alpha * log_pi + self.lagrange_multiplier * constraint_value).mean() 

                    #policy_loss = (self.alpha*log_pi - q_new_actions).mean()  ##commented 06.02.2025
                    self.policy_optimizer[i].zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer[i].step()

                    
                    #added 06.02.2025
                    # **Modify Lagrange Multiplier Update Here**
                    lambda_loss = -self.lagrange_multiplier * constraint_value.mean() * 0.01  # Ensure gradual updates

                    self.lagrange_optimizer.zero_grad()
                    lambda_loss.backward()
                    self.lagrange_optimizer.step()

                    # Soft Updates
                    for target_param, param in zip(self.target_soft_q_net1[i].parameters(), self.soft_q_net1[i].parameters()):
                        target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

                    for target_param, param in zip(self.target_soft_q_net2[i].parameters(), self.soft_q_net2[i].parameters()):
                        target_param.data.copy_(target_param.data*(1.0 - self.tau) + param.data*self.tau)

            else:
                pass

    def predict(self, observations: List[List[float]], deterministic: bool = None):
        r"""Provide actions for current time step.

        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` <= :attr:`time_step` 
        else will use policy to sample actions.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[float]
            Action values
        """

        deterministic = False if deterministic is None else deterministic

        if self.time_step > self.end_exploration_time_step or deterministic:
            actions = self.get_post_exploration_prediction(observations, deterministic)
            
        else:
            actions = self.get_exploration_prediction(observations)

        self.actions = actions
        self.next_time_step()
        return actions

    def get_post_exploration_prediction(self, observations: List[List[float]], deterministic: bool) -> List[List[float]]:
        """Action sampling using policy, post-exploration time step"""

        actions = []

        for i, o in enumerate(observations):
            o = self.get_encoded_observations(i, o)
            o = self.get_normalized_observations(i, o)
            o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
            result = self.policy_net[i].sample(o)
            a = result[2] if deterministic else result[0]
            actions.append(a.detach().cpu().numpy()[0])

        return actions
            
    def get_exploration_prediction(self, observations: List[List[float]]) -> List[List[float]]:
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`."""

        # random actions
        return [list(self.action_scaling_coefficient*s.sample()) for s in self.action_space]

    def get_normalized_reward(self, index: int, reward: float) -> float:
        return (reward - self.r_norm_mean[index])/self.r_norm_std[index]

    def get_normalized_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]:
        try:
            return (np.array(observations, dtype = float) - self.norm_mean[index])/self.norm_std[index]
        except:
            # self.time_step >= self.standardize_start_time_step and self.batch_size <= len(self.replay_buffer[i])
            logging.debug('obs:',observations)
            logging.debug('mean:',self.norm_mean[index])
            logging.debug('std:',self.norm_std[index])
            logging.debug(self.time_step, self.standardize_start_time_step, self.batch_size, len(self.replay_buffer[0]))
            assert False

    #def get_encoded_observations(self, index: int, observations: List[float]) -> npt.NDArray[np.float64]: #commented 07.02.2025 due to an error
        #return np.array([j for j in np.hstack(self.encoders[index]*np.array(observations, dtype=float)) if j != None], dtype = float)


    def get_encoded_observations(self, index: int, observations: Union[dict, List[Any]]) -> np.ndarray:
    # Build the observation list using keys and corresponding encoders.
        obs_list = []
        if isinstance(observations, dict):
            print("Raw observation dictionary:", observations)
            # Here self.observation_names[index] is assumed to be the list of expected keys for this observation.
            for encoder, key in zip(self.encoders[index], self.observation_names[index]):
                val = observations.get(key, None)
                if val is None:
                    logging.warning(f"Key '{key}' is missing in observations. Using default for encoder {encoder.__class__.__name__}.")
                    # If the encoder has defined classes, use the first one as a default; otherwise use 0.0.
                    default_val = encoder.classes[0] if hasattr(encoder, "classes") and len(encoder.classes) > 0 else 0.0
                    obs_list.append(default_val)
                else:
                    if isinstance(val, dict):
                        if "value" in val and val["value"] is not None:
                            obs_list.append(val["value"])
                        else:
                            logging.warning(f"Observation for key '{key}' is a dict but missing a valid 'value'. Using default.")
                            default_val = encoder.classes[0] if hasattr(encoder, "classes") and len(encoder.classes) > 0 else 0.0
                            obs_list.append(default_val)
                    else:
                        obs_list.append(val)
        else:
            # If observations is a list, assume the order matches that of self.encoders[index].
            for encoder, val in zip(self.encoders[index], observations):
                if val is None:
                    logging.warning("Observation value is None; using default.")
                    default_val = encoder.classes[0] if hasattr(encoder, "classes") and len(encoder.classes) > 0 else 0.0
                    obs_list.append(default_val)
                else:
                    if isinstance(val, dict):
                        if "value" in val and val["value"] is not None:
                            obs_list.append(val["value"])
                        else:
                            logging.warning("Observation dict missing 'value'; using default.")
                            default_val = encoder.classes[0] if hasattr(encoder, "classes") and len(encoder.classes) > 0 else 0.0
                            obs_list.append(default_val)
                    else:
                        obs_list.append(val)
        
        #print("Intermediate obs_list:", obs_list) ##########
        
        try:
            # Multiply the list of encoders with the numeric observation array.
            # (This relies on each encoder having an overloaded __mul__ operator that applies the encoding.)
            encoded = self.encoders[index] * np.array(obs_list, dtype=float)
            #print("After encoder multiplication, raw encoded value:", encoded) ######
        except Exception as e:
            print("Error during encoder multiplication:", e)
            raise
    
        # Flatten the encoded output and filter out any None values.
        flat_encoded = []
        for item in np.hstack(encoded):
            if item is None:
                continue
            elif isinstance(item, (list, np.ndarray)):
                flat_encoded.extend(np.array(item, dtype=float).flatten().tolist())
            else:
                flat_encoded.append(item)
        
        final_encoded = np.array(flat_encoded, dtype=float)
        #print("Final encoded observation:", final_encoded)  ######
        return final_encoded








    def set_networks(self, internal_observation_count: int = None):
        internal_observation_count = 0 if internal_observation_count is None else internal_observation_count

        for i in range(len(self.action_dimension)):
            observation_dimension = self.observation_dimension[i] + internal_observation_count
            # init networks
            self.soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.target_soft_q_net1[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)
            self.target_soft_q_net2[i] = SoftQNetwork(observation_dimension, self.action_dimension[i], self.hidden_dimension).to(self.device)

            for target_param, param in zip(self.target_soft_q_net1[i].parameters(), self.soft_q_net1[i].parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_soft_q_net2[i].parameters(), self.soft_q_net2[i].parameters()):
                target_param.data.copy_(param.data)

            # Policy
            self.policy_net[i] = PolicyNetwork(observation_dimension, self.action_dimension[i], self.action_space[i], self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
            self.soft_q_optimizer1[i] = optim.Adam(self.soft_q_net1[i].parameters(), lr=self.lr)
            self.soft_q_optimizer2[i] = optim.Adam(self.soft_q_net2[i].parameters(), lr=self.lr)
            self.policy_optimizer[i] = optim.Adam(self.policy_net[i].parameters(), lr=self.lr)
            self.target_entropy[i] = -np.prod(self.action_space[i].shape).item()

    def set_encoders(self) -> List[List[Encoder]]:
        encoders = super().set_encoders()

        for i, o in enumerate(self.observation_names):
            for j, n in enumerate(o):
                if n == 'net_electricity_consumption':
                    encoders[i][j] = RemoveFeature()
            
                else:
                    pass

        return encoders

class SACRBC(SAC):
    r"""Uses :py:class:`citylearn.agents.rbc.RBC` to select actions during exploration before using :py:class:`citylearn.agents.sac.SAC`.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment.
    rbc: RBC
        :py:class:`citylearn.agents.rbc.RBC` or child class, used to select actions during exploration.
    
    Other Parameters
    ----------------
    **kwargs : Any
        Other keyword arguments used to initialize super class.
    """
    
    def __init__(self, env: CityLearnEnv, rbc: Union[RBC, str] = None, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.__set_rbc(rbc, **kwargs)

    @property
    def rbc(self) -> RBC:
        """:py:class:`citylearn.agents.rbc.RBC` class child class or string path to an RBC 
        class e.g. 'citylearn.agents.rbc.RBC', used to select actions during exploration."""

        return self.__rbc
    
    def __set_rbc(self, rbc: RBC, **kwargs):
        if rbc is None:
            rbc = RBC(self.env, **kwargs)
        
        elif isinstance(rbc, RBC):
            pass

        elif isinstance(rbc, str):
            rbc = self.env.load_agent(rbc, env=self.env, **kwargs)

        else:
            rbc = rbc(self.env, **kwargs)
        
        self.__rbc = rbc

    def get_exploration_prediction(self, observations: List[float]) -> List[float]:
        """Return actions using :class:`RBC`."""

 

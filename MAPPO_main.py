import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For setting the style
import csv
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO_MPE
import os
from IPython import display as ipy_display
from matplotlib.ticker import FuncFormatter
from pettingzoo.mpe import simple_spread_v3
from PIL import Image, ImageDraw, ImageFont
import io

class Runner_MAPPO_PettingZoo:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set seaborn style
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # Create PettingZoo environment
        self.env = simple_spread_v3.parallel_env(
            N=3,  # Number of agents
            local_ratio=0.5,
            max_cycles=args.episode_limit,  # Episode limit
            continuous_actions=False,
            render_mode=None
        )
        
        # Reset environment to get initial observations
        observations, infos = self.env.reset(seed=self.seed)
        
        # Get agent information
        self.agent_list = list(self.env.agents)
        self.args.N = len(self.agent_list)  # Number of agents
        
        # Get observation and action dimensions
        self.args.obs_dim_n = []
        self.args.action_dim_n = []
        
        for agent in self.agent_list:
            obs_space = self.env.observation_space(agent)
            action_space = self.env.action_space(agent)
            
            self.args.obs_dim_n.append(obs_space.shape[0])
            self.args.action_dim_n.append(action_space.n)
        
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print("Number of agents:", self.args.N)
        print("Agent list:", self.agent_list)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(
            self.env_name, self.number, self.seed))

        # Lists for evaluation rewards and steps
        self.evaluate_rewards = []
        self.eval_steps = []
        self.total_steps = 0
        
        # GIF saving variables
        self.gif_save_freq = 20000  # Save GIF every 20k steps
        self.last_gif_save = 0  # Track when last GIF was saved

        # Create folder for saving data if it doesn't exist
        os.makedirs('./data_train', exist_ok=True)
        os.makedirs('./gifs', exist_ok=True)  # Create folder for GIFs

        # Set up live plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        (self.line,) = self.ax.plot([], [], color='orange', label='MAPPO')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Episode Reward')
        self.ax.set_title('Simple Spread')
        self.ax.legend(loc='lower right')
        self.fig.show()

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1  # Number of evaluations performed
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1
            
            # Check if it's time to save GIF
            if self.total_steps - self.last_gif_save >= self.gif_save_freq:
                self.save_gif_episode()
                self.last_gif_save = self.total_steps

            _, episode_steps = self.run_episode_pettingzoo(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()  # Final evaluation
        # Save final GIF
        self.save_gif_episode()
        self.env.close()
        self.save_eval_csv()
        plt.ioff()
        plt.show()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_pettingzoo(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward /= self.args.evaluate_times

        # Record evaluation steps and rewards
        self.eval_steps.append(self.total_steps)
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)

        # Save the model if necessary
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

        self.save_eval_csv()
        self.plot_eval_rewards()

    def save_eval_csv(self):
        csv_filename = './data_train/MAPPO_env_{}_number_{}_seed_{}.csv'.format(
            self.env_name, self.number, self.seed)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Training Steps', 'Evaluation Reward'])
            for step, reward in zip(self.eval_steps, self.evaluate_rewards):
                writer.writerow([step, reward])

    def plot_eval_rewards(self):
        # Update plot data
        self.line.set_xdata(self.eval_steps)
        self.line.set_ydata(self.evaluate_rewards)
        self.ax.relim()
        self.ax.autoscale_view()

        # Dynamic formatter for the X-axis
        def dynamic_formatter(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{int(x)}'
        self.ax.xaxis.set_major_formatter(FuncFormatter(dynamic_formatter))

        # Redraw canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save the plot image
        plt.savefig('./data_train/MAPPO_env_{}_number_{}_seed_{}_eval.png'.format(
            self.env_name, self.number, self.seed))

    def get_agent_positions(self, render_env):
        """Extract agent positions from the environment state"""
        positions = []
        
        # Get world state from the environment
        world = render_env.unwrapped.world
        
        # Extract agent positions
        for i, agent in enumerate(world.agents):
            # Convert world coordinates to pixel coordinates
            # The environment typically uses coordinates from -1 to 1
            # We need to scale them to the render size
            x_world = agent.state.p_pos[0]
            y_world = agent.state.p_pos[1]
            
            # Scale to render dimensions (assuming 700x700 render size)
            # This may need adjustment based on actual render size
            render_size = 700
            x_pixel = int((x_world + 1) * render_size / 2)
            y_pixel = int((1 - y_world) * render_size / 2)  # Flip Y axis
            
            positions.append((x_pixel, y_pixel))
        
        return positions

    def add_colored_agent_markers(self, frame, agent_positions):
        """Add colored circular markers with numbers on top of agents"""
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Agent colors: Agent 1 = Blue, Agent 2 = Orange, Agent 3 = Green
        agent_colors = [
            (0, 100, 255),      # Blue for Agent 1
            (255, 165, 0),      # Orange for Agent 2  
            (0, 200, 0)         # Green for Agent 3
        ]
        
        # Border colors (darker versions for better visibility)
        border_colors = [
            (0, 50, 150),       # Dark Blue
            (200, 100, 0),      # Dark Orange
            (0, 120, 0)         # Dark Green
        ]
        
        # Marker settings
        marker_radius = 28  # Radius of the colored circle (much larger for better visibility)
        border_width = 4    # Width of the border
        
        # Try to get a good font for numbers - INCREASED FONT SIZE
        try:
            font = ImageFont.truetype("arial.ttf", 48)  # Increased from 36 to 48
        except:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)  # Increased from 36 to 48
            except:
                try:
                    font = ImageFont.load_default()
                    # Try to get a larger default font if possible
                    font = font.font_variant(size=48)
                except:
                    font = None
        
        for i, (x, y) in enumerate(agent_positions):
            if i < len(agent_colors):
                # Draw border circle (slightly larger)
                border_color = border_colors[i % len(border_colors)]
                draw.ellipse([
                    x - marker_radius - border_width, 
                    y - marker_radius - border_width,
                    x + marker_radius + border_width, 
                    y + marker_radius + border_width
                ], fill=border_color)
                
                # Draw main colored circle
                agent_color = agent_colors[i % len(agent_colors)]
                draw.ellipse([
                    x - marker_radius, 
                    y - marker_radius,
                    x + marker_radius, 
                    y + marker_radius
                ], fill=agent_color)
                
                # Draw the agent number in the center
                agent_number = str(i + 1)  # Agent numbers 1, 2, 3
                
                # Get text size for centering
                if font:
                    bbox = draw.textbbox((0, 0), agent_number, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    # Account for font baseline offset
                    text_x = x - text_width // 2
                    text_y = y - text_height // 2 - bbox[1] // 2
                else:
                    # Estimate larger text size if font is not available
                    text_width = 24  # Increased from 18 to 24
                    text_height = 32  # Increased from 24 to 32
                    # Center the text
                    text_x = x - text_width // 2
                    text_y = y - text_height // 2
                
                # Draw text with white color for good contrast and add black outline for better visibility
                if font:
                    # Draw black outline for better visibility
                    outline_size = 2
                    for dx in range(-outline_size, outline_size + 1):
                        for dy in range(-outline_size, outline_size + 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), agent_number, fill=(0, 0, 0), font=font)
                    # Draw white text on top
                    draw.text((text_x, text_y), agent_number, fill=(255, 255, 255), font=font)
                else:
                    # Fallback without font - draw with better centering
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                draw.text((text_x, text_y), agent_number, fill=(255, 255, 255))
                            else:
                                draw.text((text_x + dx, text_y + dy), agent_number, fill=(0, 0, 0))
        
        return np.array(img)

    def save_gif_episode(self):
        """Save a GIF of one episode using the current policy with colored agent markers"""
        print(f"Saving GIF at step {self.total_steps}...")
        
        # Create a separate environment for rendering
        render_env = simple_spread_v3.parallel_env(
            N=3,
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False,
            render_mode='rgb_array'  # Enable rendering
        )
        
        frames = []
        observations, infos = render_env.reset(seed=self.seed)
        
        # Capture initial frame with colored markers
        frame = render_env.render()
        if frame is not None:
            try:
                agent_positions = self.get_agent_positions(render_env)
                colored_frame = self.add_colored_agent_markers(frame, agent_positions)
                frames.append(Image.fromarray(colored_frame))
            except Exception as e:
                print(f"Warning: Could not add colored markers to frame: {e}")
                frames.append(Image.fromarray(frame))
        
        episode_step = 0
        
        while render_env.agents and episode_step < self.args.episode_limit:
            # Convert observations to list format
            obs_n = [observations[agent] for agent in self.agent_list if agent in observations]
            
            if len(obs_n) != self.args.N:
                break
            
            # Get actions using current policy (evaluation mode)
            a_n, _ = self.agent_n.choose_action(obs_n, evaluate=True)
            
            # Create action dictionary
            actions = {}
            for i, agent in enumerate(self.agent_list):
                if agent in render_env.agents:
                    actions[agent] = a_n[i]
            
            # Step environment
            observations, rewards, terminations, truncations, infos = render_env.step(actions)
            
            # Capture frame with colored markers
            frame = render_env.render()
            if frame is not None:
                try:
                    agent_positions = self.get_agent_positions(render_env)
                    colored_frame = self.add_colored_agent_markers(frame, agent_positions)
                    frames.append(Image.fromarray(colored_frame))
                except Exception as e:
                    print(f"Warning: Could not add colored markers to frame: {e}")
                    frames.append(Image.fromarray(frame))
            
            episode_step += 1
            
            # Check if episode should end
            if all(terminations.values()) or all(truncations.values()):
                break
        
        render_env.close()
        
        # Save GIF
        if frames:
            gif_filename = f'./gifs/MAPPO_env_{self.env_name}_step_{self.total_steps}.gif'
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # Duration per frame in milliseconds
                loop=0
            )
            print(f"GIF saved: {gif_filename}")
        else:
            print("No frames captured for GIF")

    def run_episode_pettingzoo(self, evaluate=False):
        episode_reward = 0
        observations, infos = self.env.reset(seed=self.seed)
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
            
        episode_step = 0
        
        while self.env.agents:  # PettingZoo style episode loop
            # Convert observations to list format for compatibility with MAPPO
            obs_n = [observations[agent] for agent in self.agent_list if agent in observations]
            
            # If not all agents are active, break
            if len(obs_n) != self.args.N:
                break
            
            # Get actions from agents
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            
            # Global state as concatenation of all observations
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            
            # Create action dictionary for PettingZoo
            actions = {}
            for i, agent in enumerate(self.agent_list):
                if agent in self.env.agents:
                    actions[agent] = a_n[i]
            
            # Step environment
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Convert rewards to list format
            r_n = [rewards[agent] for agent in self.agent_list if agent in rewards]
            
            # Convert done flags to list format
            done_n = [terminations[agent] or truncations[agent] for agent in self.agent_list if agent in terminations]
            
            # Sum rewards for episode reward (you might want to change this based on your needs)
            if r_n:
                episode_reward += np.sum(r_n)

            if not evaluate and len(r_n) == self.args.N:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            episode_step += 1
            
            # Check if episode should end
            if all(terminations.values()) or all(truncations.values()) or episode_step >= self.args.episode_limit:
                break

        # Store final value for advantage calculation
        if not evaluate and len(obs_n) == self.args.N:
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step, v_n)

        return episode_reward, episode_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in PettingZoo Spread environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=50, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Number of evaluations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Number of neurons in RNN hidden layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Number of neurons in MLP hidden layers")
    parser.add_argument("--alliance_hidden_dim", type=int, default=64, help="Number of neurons in alliance hidden layers")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Use advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Use reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Use reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Use learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Use gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Use orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU (if False, use tanh)")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=bool, default=False, help="Whether to add agent id")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Whether to use value clipping")

    args = parser.parse_args()
    runner = Runner_MAPPO_PettingZoo(args, env_name="simple_spread", number=1, seed=2)
    runner.run()
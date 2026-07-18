# Agent : Gambler
# Environment: return p(r,s'|s,a) 
# State : capital (0, 1, 2, ..., 100)
# Action : bet (1, 2, ..., min(s, 100-s))
# Reward : 0 for all transitions except for the transition to state 100, which has reward 1.

def reward(s, action):
    if s + action == 100:
        return 1

    return 0

def policy_evaluation(policy, p_h, num_iter):
    v = [0.0] * 101
    for _ in range(num_iter):
        new_v = v.copy()
        for s in range(1, 100):
            action = policy[s]
            new_v[s] = (
                p_h * (reward(s, action) + v[s+action])
                + (1-p_h) * (reward(s, -action) + v[s-action])
            )
        v = new_v

    return v

def policy_improvement(v, p_h):
    policy = [0] * 101
    for s in range(1, 100):
        best_action = 0
        best_value = float('-inf')
        for action in range(1, min(s, 100-s)+1):
            action_value = (
                p_h * (reward(s, action) + v[s+action])
                + (1-p_h) * (reward(s, -action) + v[s-action])
            )
            if action_value > best_value:
                best_value = action_value
                best_action = action

        policy[s] = best_action

    return policy
def policy_iteration(p_h, num_eval_iter, num_policy_iter):
    policy = [0] * 101
    for _ in range(num_policy_iter):
        v = policy_evaluation(policy, p_h, num_eval_iter)
        policy = policy_improvement(v, p_h)

    return policy, v

def value_iteration(p_h,num_iter):
    value_iter = []
    delta_iter = []
    v = [0.0] * 101

    for _ in range(num_iter):
        delta = 0
        for s in range(1,100):
            old_value = v[s]
            action_values = []
            for action in range(1, min(s, 100-s)+1):
                action_value = (
                    p_h * (reward(s, action) + v[s+action])
                    + (1-p_h) * (reward(s, -action) + v[s-action])
                )
                action_values.append(action_value)

            v[s] = max(action_values)
            delta = max(delta, abs(old_value - v[s]))

        delta_iter.append(delta)
        value_iter.append(v.copy())

    return value_iter, delta_iter


def optimal_policy(v, p_h):
    policy = [0] * 101
    for s in range(1, 100):
        best_action = 0
        best_value = float('-inf')
        for action in range(1, min(s, 100-s)+1):
            action_value = (
                p_h * (reward(s, action) + v[s+action])
                + (1-p_h) * (reward(s, -action) + v[s-action])
            )
            if action_value > best_value:
                best_value = action_value
                best_action = action

        policy[s] = best_action

    return policy


if __name__ == "__main__":
    p_h = 0.4
    num_iter = 50
    value_iter, delta_iter = value_iteration(p_h, num_iter)
    final_policy = optimal_policy(value_iter[-1], p_h)

    # draw the value iteration, each step is a line with a color
    # draw delta iter, delta_iter is a list of delta for each iteration, draw it as a line plot
    import matplotlib.pyplot as plt
    for i in range(num_iter):
        plt.plot(range(1, 100), value_iter[i][1:100], label=f'iteration {i}')
    plt.xlabel('Capital')
    plt.ylabel('Value')
    plt.title('Value Iteration for Gambler Problem')
    plt.legend()
    plt.show()

    # draw delta iteration
    plt.figure()
    plt.plot(delta_iter)
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.title('Convergence of Value Iteration')
    plt.show()

    # draw policy from the final iteration
    plt.figure()
    plt.bar(range(1, 100), final_policy[1:100])
    plt.xlabel('Capital')
    plt.ylabel('Final policy: stake')
    plt.title('Final Policy for Gambler Problem')
    plt.show()


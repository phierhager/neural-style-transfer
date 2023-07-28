from deep_learning.functions import preprocess, load_data, plot_costs, two_layer_model, L_layer_model, predict


def run_two_layer_model():
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    learning_rate = 0.01
    num_iterations = 1000

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x = preprocess(train_x_orig)
    test_x = preprocess(test_x_orig)
    parameters, costs = two_layer_model(train_x, train_y, layers_dims=(
        n_x, n_h, n_y), learning_rate=learning_rate, num_iterations=num_iterations, print_cost=True)
    plot_costs(costs, learning_rate)
    pred_test = predict(test_x, test_y, parameters)


def run_four_layer_model():
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
    learning_rate = 0.01
    num_iterations = 1000

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x = preprocess(train_x_orig)
    test_x = preprocess(test_x_orig)
    parameters, costs = L_layer_model(
        train_x, train_y, layers_dims, learning_rate=learning_rate, num_iterations=num_iterations, print_cost=True)
    pred_test = predict(test_x, test_y, parameters)
    plot_costs(costs, learning_rate)

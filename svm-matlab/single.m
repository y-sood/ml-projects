% Load dataset
data = readtable("C:\Users\Tejaswi S\Documents\yvs\period2\optimization\ass2\svm\bcwd\wdbc_modifiedlat.csv");
labels = data{:, 2};
features = data{:, 3:end};

% Train/test split (80/20)
train_features = features(1:456, :);
train_labels = labels(1:456);
test_features = features(457:end, :);
test_labels = labels(457:end, :);

X = train_features;
y = train_labels;

[m, n] = size(X);

% Fixed matrices for all C values
H = [eye(n), zeros(n, m + 1);  
     zeros(m + 1, n + m + 1)];

A_margin = [-y .* X, -y, -eye(m)];
A_slack  = [zeros(m, n + 1), eye(m)];
A = [A_margin; A_slack];

b = [ones(m, 1); zeros(m, 1)];

% Containers to store results
acc = []; sens = []; spec = [];
tp = []; fn = []; tn = []; fp = [];

C_values = [100, 200, 300, 500, 700, 900, 1000, 1200];

for i = 1:length(C_values)
    C = C_values(i);
    f = [zeros(n, 1); 0; C * ones(m, 1)];
    
    options = optimoptions('quadprog', 'Display', 'off');
    [sol, ~] = quadprog(H, f, A, b);
    
    w = sol(1:n);
    bias = sol(n+1);
    
    predictions = sign(test_features * w + bias);
    
    % Confusion matrix components
    tp_ = sum((predictions == 1) & (test_labels == 1));
    fp_ = sum((predictions == 1) & (test_labels == -1));
    tn_ = sum((predictions == -1) & (test_labels == -1));
    fn_ = sum((predictions == -1) & (test_labels == 1));
    
    % Metrics
    acc_ = (tp_ + tn_) / length(test_labels);
    sens_ = tp_ / (tp_ + fn_);
    spec_ = tn_ / (tn_ + fp_);
    
    % Store
    acc = [acc, acc_];
    sens = [sens, sens_];
    spec = [spec, spec_];
    tp = [tp, tp_];
    tn = [tn, tn_];
    fp = [fp, fp_];
    fn = [fn, fn_];
end

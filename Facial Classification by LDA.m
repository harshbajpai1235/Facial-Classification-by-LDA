clc; clear; close all;
% Loading Dataset
femaleDir = '/Users/harshbajpai/Desktop/FaceClassification_Data/Female';
maleDir = '/Users/harshbajpai/Desktop/FaceClassification_Data/Male';
imgSize = [32, 32];
femFiles = dir(fullfile(femaleDir, '*.tif'));
malFiles = dir(fullfile(maleDir, '*.tif'));
numFem = numel(femFiles);
numMal = numel(malFiles);
numTotal = numFem + numMal;
X = zeros(numTotal, prod(imgSize));
labels = [ones(numFem,1); -ones(numMal,1)];
% Loading Female images
for i = 1:numFem
img = imread(fullfile(femaleDir, femFiles(i).name));
img = imresize(rgb2grayIfNeeded(img), imgSize);
X(i,:) = double(img(:)');
end
% Loading Male images
for i = 1:numMal
img = imread(fullfile(maleDir, malFiles(i).name));
img = imresize(rgb2grayIfNeeded(img), imgSize);
X(numFem + i,:) = double(img(:)');
end
% PCA
meanFace = mean(X, 1);
X_centered = X - meanFace;
covarianceMatrix = (X_centered' * X_centered) / (numTotal - 1);
[U, D] = eig(covarianceMatrix);
[eigvals, sortIdx] = sort(diag(D), 'descend');
U = U(:, sortIdx);
% choosing K PCA dimension
K = 37;
projPCA = U(:, 1:K);
Z = X_centered * projPCA;
% LDA
Z_female = Z(labels == 1, :);
Z_male = Z(labels == -1, :);
mu_female = mean(Z_female, 1);
mu_male = mean(Z_male, 1);
mu_total = mean(Z, 1);
% Between-class scatter
Sb = numFem * (mu_female - mu_total)' * (mu_female - mu_total) + ...
numMal * (mu_male - mu_total)' * (mu_male - mu_total);
% Within-class scatter
Sw = zeros(K, K);
for i = 1:size(Z_female,1)
diff = Z_female(i,:) - mu_female;
Sw = Sw + (diff' * diff);
end
for i = 1:size(Z_male,1)
diff = Z_male(i,:) - mu_male;
Sw = Sw + (diff' * diff);
end
Sw = Sw + 1e-6 * eye(K); % Regularization
% Solving LDA
[W_lda, lambda] = eig(Sb, Sw);
[~, maxIdx] = max(diag(lambda));
ldaVector = W_lda(:, maxIdx);
ldaScores = Z * ldaVector;
% Classify with Threshold
optimalThreshold = -0.005; % Found via grid search in reference
predictions = sign(ldaScores - optimalThreshold);
accuracy = mean(predictions == labels) * 100;
% Output
fprintf('Final PCA dimension K = %d\n', K);
fprintf('Optimal LDA threshold = %.4f\n', optimalThreshold);
fprintf('Classification Accuracy = %.2f%%\n', accuracy);
% Plotting LDA Score Distribution
figure;
histogram(ldaScores(labels==1), 'FaceColor', '#1f77b4'); hold on;
histogram(ldaScores(labels==-1), 'FaceColor', '#d62728');
xline(optimalThreshold, '--k', 'LineWidth', 1.5);
legend('Female', 'Male', 'Threshold');
xlabel('LDA Projection Score'); ylabel('Frequency');
title('LDA Score Distribution by Class');
% Helper Function
function gray = rgb2grayIfNeeded(img)
if ndims(img) == 3
gray = rgb2gray(img);
else
gray = img;
end
end
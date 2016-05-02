
% load '../output_data/train_acc_plain_56layer.txt';
% load '../output_data/test_acc_plain_56layer.txt';
% 
% plain_x1 = train_acc_plain_56layer(:,1)/1e4;
% plain_y1 = train_acc_plain_56layer(:,2);
% plain_x2 = test_acc_plain_56layer(:,1)/1e4;
% plain_y2 = test_acc_plain_56layer(:,2);
% 
% figure(1);
% hold on;
% plot(plain_x1, plain_y1, 'r-');
% plot(plain_x2, plain_y2, 'b-');
% axis( [ 0 7 0 0.2 ] )
% xlabel('iteration (1e4)');
% ylabel('test & train error(ratio)' );
% title('plain 56 layer result' );
% grid on;
% hold off;
% 
% load '../output_data/train_acc_res_56layer.txt';
% load '../output_data/test_acc_res_56layer.txt';
% 
% res_x1 = train_acc_res_56layer(:,1)/1e4;
% res_y1 = train_acc_res_56layer(:,2);
% res_x2 = test_acc_res_56layer(:,1)/1e4;
% res_y2 = test_acc_res_56layer(:,2);
% 
% figure(2);
% hold on;
% plot(res_x1, res_y1, 'r-');
% plot(res_x2, res_y2, 'b-');
% axis( [ 0 7 0 0.2 ] )
% xlabel('iteration (1e4)');
% ylabel('test & train error(ratio)' );
% title('residual 56 layer result' );
% grid on;
% hold off;


load '../output_data/train_acc_res_110layer.txt';
load '../output_data/test_acc_res_110layer.txt';
load '../output_data/train_acc_bottle_164layer.txt';
load '../output_data/test_acc_bottle_164layer.txt';

res_x1 = train_acc_res_110layer(:,1)/1e4;
res_y1 = train_acc_res_110layer(:,2);
res_x2 = test_acc_res_110layer(:,1)/1e4;
res_y2 = test_acc_res_110layer(:,2);

res_x3 = train_acc_bottle_164layer(:,1)/1e4;
res_y3 = train_acc_bottle_164layer(:,2);
res_x4 = test_acc_bottle_164layer(:,1)/1e4;
res_y4 = test_acc_bottle_164layer(:,2);

figure(3);
hold on;
plot(res_x1, res_y1, 'r.');
plot(res_x2, res_y2, 'r-');
plot(res_x3, res_y3, 'b.');
plot(res_x4, res_y4, 'b-');
axis( [ 0 7 0 0.2 ] )
xlabel('iteration (1e4)');
ylabel('test & train error(ratio)' );
title('post-active 110 vs pre-active 164 layer' );
grid on;
hold off;





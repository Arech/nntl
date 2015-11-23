function mnist2bin( fname, cnt, cntt )
%MNIST2BIN loads MNIST data from DeepLearnToolbox's mnist_uint8.mat and
% stores it into file fname in nntl bin format (see nntl/_supp/binfile.h)
% cnt and cntt allows to specify how many samples to take from training and
% testing set respectively.

if ~exist('cnt','var'), cnt=0; end
if ~exist('cntt','var'), cntt=0; end

load mnist_uint8;
train_x = double(train_x) / 255;

if 0>=cnt || cnt>size(train_x,1)
	cnt = size(train_x,1);
else
	train_x = train_x(1:cnt,:);
end
train_y = double(train_y(1:cnt,:));

if 0<=cntt || cntt>size(test_x,1), cntt = size(test_x,1); end
test_x  = double(test_x(1:cntt,:))  / 255;
test_y  = double(test_y(1:cntt,:));


td=struct('train_x',train_x,'train_y',train_y,'test_x',test_x,'test_y',test_y);

export_2bin(td,fname);

end


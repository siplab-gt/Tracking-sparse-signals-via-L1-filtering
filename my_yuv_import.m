function [Y,U,V]=my_yuv_import(filename,dims,numfrm,varargin)
%Imports YUV sequence
%[Y,U,V]=yuv_import(filename,dims,numfrm,startfrm)
%Version: 2.10, Date: 2007/10/01, author: Nikola Sprljan
%
%Input:
% filename - YUV sequence file
% dims - dimensions of the frame [width height]
% numfrm - number of frames to read
% varargin - input argument [optional]
%               - 'start' - [optional, default = 0] specifies from which frame to start reading
%                           with the convention that the first frame of the sequence is 0-
%                           numbered
%               - 'yuvformat' - [optional, default = 'YUV420_8']. YUV format, supported formats 
%                           are: 
%                           'YUV444_8' = 4:4:4 sampling, 8-bit precision 
%                           'YUV420_8' = 4:2:0 sampling, 8-bit precision
%                           'YUV420_16' = 4:2:0 sampling, 16-bit precision
%               - 'output' - [optional, default = 'cell']. Possible outputs
%                             'cell'
%                             'mat'
%                           
%
% Output:
% Y, U ,V - cell arrays or matrices of Y, U and V components  
%
%Note:
% Supported YUV formats are (corresponding yuvformat variable):
%  'YUV420_8' = 4:2:0 sampling, 8-bit precision (default)
%  'YUV420_16' = 4:2:0 sampling, 16-bit precision
%
%Example:
% [Y, U, V] = yuv_import('FOREMAN_352x288_30_orig_01.yuv',[352 288],2);
% image_show(Y{1},256,1,'Y component');
% [Y, U, V] = yuv_import('sequence.yuv',[1920 1080],2,'start',0,'yuvformat','YUV420_16');

p=inputParser;
p.addParamValue('start',0);
p.addParamValue('yuvformat','YUV420_8');
p.addParamValue('output','cell');

if nargin==3
    p.addRequired('filename', @(x)fopen(x,'r')>0);
    p.addRequired('dims');
    p.addRequired('numfrm');
    p.parse(filename,dims,numfrm);
elseif nargin>3
    p.addRequired('filename', @(x)fopen(x,'r')>0);
    p.addRequired('dims');
    p.addRequired('numfrm');
    p.parse(filename,dims,numfrm,varargin{:});
else
    error('Missing required arguments: filename, dims, numfrm');
end
config = p.Results;

fid=fopen(filename,'r');
sampl = 420;

startfrm = config.start;
yuvformat = config.yuvformat;
inprec = 'ubit8';
if (strcmp(yuvformat,'YUV420_16'))
    inprec = 'uint16'; %'ubit16=>uint16'
elseif (strcmp(yuvformat,'YUV444_8'))
    sampl = 444;
end;

if (sampl == 420)
    dimsUV = dims / 2;
else
    dimsUV = dims;
end;
Yd = zeros(dims);
UVd = zeros(dimsUV);
frelem = numel(Yd) + 2*numel(UVd);

fseek(fid, startfrm * frelem , 0); %go to the starting frame

if strcmp(config.output,'cell')
    Y = cell(1,numfrm);
    U = cell(1,numfrm);
    V = cell(1,numfrm);
    for i=1:numfrm
        Yd = fread(fid,dims,inprec);
        Y{i} = Yd';   
        UVd = fread(fid,dimsUV,inprec);
        U{i} = UVd';
        UVd = fread(fid,dimsUV,inprec);
        V{i} = UVd';    
    end;
elseif strcmp(config.output,'mat')
    Y = zeros([flipud(dims(:))',numfrm]);
    U = zeros([flipud(dimsUV(:))',numfrm]);
    V = zeros([flipud(dimsUV(:))',numfrm]);
    for i=1:numfrm
        Yd = fread(fid,dims,inprec);
        Y(:,:,i) = Yd';   
        UVd = fread(fid,dimsUV,inprec);
        U(:,:,i) = UVd';
        UVd = fread(fid,dimsUV,inprec);
        V(:,:,i) = UVd';    
    end;
end
fclose(fid);
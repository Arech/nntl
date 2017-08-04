function [ s ] = layer_type2string( t )
%LAYER_TYPE2STRING Summary of this function goes here
%   Detailed explanation goes here

LT=static_layer_types();
switch(t)	
	case LT.lfc
		s='lfc';
		
	case LT.lph
		s='lph';
		
	case LT.lpv
		s='lpv';
		
	case LT.lid
		s='lid';
		
	case LT.src
		%error('Cant process src type here!');
		s='src';
		
	case LT.unk
		error('Invalid layer type!');
		
	case LT.custom
		s='custom';
		
	otherwise
		error(['Unknown layer type=' num2str(t,0)]);
end

end


function [t] = string2layer_type(str)
LT=static_layer_types();
switch(lower(str))
	case 'lfc'
		t=LT.lfc;
	
	case 'lph' %,'lph_decov'}
		t=LT.lph;
		
	case 'lpv'
		t=LT.lpv;
		
	case 'lid'
		t = LT.lid;
		
	otherwise
		%error(['Unknown layer type=' str]);	
		t=LT.custom;
end
end
function [ la ] = string2lossAddendum( str )
%STRING2LOSSADDENDUM Parses comma separated string and returns loss_addendum id's
% format:
% <opt_layer_type_spec> <DELIM> <opt_loss_addendum_spec> <DELIM> <opt_dropout_spec>
% <DELIM> is any of ' ,;_'
%

rem=str;
la={};

while ~isempty(rem)
	[tok,rem] = strtok(rem,' ,;_');
	
	switch(lower(tok))
		case 'decov'
			la=[la 'DeCov'];		
		case 'l1'
			la=[la 'L1'];		
		case 'l2'
			la=[la 'L2'];
		
		%the next strings might be found in a layer stereotype for a compound layer. They shouldn't trigger
		%the warning message
		case {'lpv','lph'}
			
		%if it specifies dropout we should stop parsing
		case {'dropout','nodropout','do','nodo'}
			break;
		
		otherwise
			warning('Unexpected token %s found while parsing string %s for loss_addendums specification. It was ignored', tok, str);
	end
end

if isempty(la)
	la=[];
else
	la={la};
end

end


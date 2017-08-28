function [ la, supportedList ] = string2lossAddendum( str, bWarnForUnknown )
%STRING2LOSSADDENDUM Parses comma separated string and returns loss_addendum id's
% format:
% <opt_layer_type_spec> <DELIM> <opt_loss_addendum_spec> <DELIM> <opt_dropout_spec>
% <DELIM> is any of ' ,;_'
%

if nargin<2
	bWarnForUnknown=true;
end
bWarnForUnknown=logical(bWarnForUnknown);

rem=str;
la={};
supportedList={'decov','l1','l2'};

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
		%case {'lpv','lph'}
			
		%if it specifies dropout we should stop parsing
		%case {'dropout','nodropout','do','nodo'}
		%	break;
		
		otherwise
			if bWarnForUnknown
				warning('Unexpected token %s found while parsing string %s for loss_addendums specification. It was ignored', tok, str);
			end
	end
end

if isempty(la)
	la=[];
%else
	%la={la};
end

end


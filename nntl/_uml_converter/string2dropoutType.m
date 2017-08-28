%obsolete!
function [ dout ] = string2dropoutType( str )
%Parses comma separated string and returns dropout type string / bool / empty
% format:
% <opt_layer_type_spec> <DELIM> <opt_loss_addendum_spec> <DELIM> <opt_dropout_spec>
% <DELIM> is any of ' ,;_'
%

error('function obsolete!');

rem=str;
dout=[];

while ~isempty(rem)
	[tok,rem] = strtok(rem,' ,;_');
	
	switch(lower(tok))
		%just skipping layer types spec and loss adds spec
		case {'lpv','lph','decov','l1','l2'}
			
		%if it specifies dropout
		case {'dropout','do'}
			dout=true;
			if ~isempty(rem)
				%trying to get suffux
				[tok,rem] = strtok(rem,' ,;_');
				assert(isempty(rem),'Dropout specification must be at the end of the string');
				assert(~isempty(tok),'WTF?');
				dout=upper(tok);
			end
			break;
			
		case {'nodropout','nodo'}
			assert(isempty(rem),'Dropout specification must be at the end of the string');
			dout=false;
			break;
		
		otherwise
			warning('Unexpected token %s found while parsing string %s for dropout specification. It was ignored', tok, str);
	end
end

end


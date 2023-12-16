function [ layerType, lossAdd, dropout, hasCustomFlag ] = parseStereotype( str )
%PARSESTEREOTYPE Parses stereotype string and extracts relevant information from it
%We're using UML packages to model compound layers such as LPH or LPH. Packages lacks attributes and many
%other means to specify its properties, however they do support stereotypes. We're using stereotypes to set
%package/layer properties.

LT=static_layer_types();

[lossAdd,supportedList] = string2lossAddendum(str,false);
dropout=[];
layerType = LT.custom;
hasCustomFlag=[];

rem=str;

while ~isempty(rem)
	[tok,rem] = strtok(rem,' ,;_');
	
	str2check = lower(tok);
	
	switch(str2check)		
		%the next strings might be found in a layer stereotype for a compound layer. They shouldn't trigger
		%the warning message
		case 'lpv'
			assert(layerType==LT.custom, ['layer type has already been specified in stereotype str=' str]);
			layerType = LT.lpv;
			
		case 'lph'
			assert(layerType==LT.custom, ['layer type has already been specified in stereotype str=' str]);
			layerType = LT.lph;
			
		%if it specifies dropout
		case {'dropout','do'}
			dropout=true;
			if ~isempty(rem)
				%trying to get suffux
				[tok,rem] = strtok(rem,' ,;_');
				assert(isempty(rem),'Dropout specification must be at the end of the string');
				assert(~isempty(tok),'WTF?');
				dropout=upper(tok);
			end
			break;
			
		case {'nodropout','nodo'}
			assert(isempty(rem),'Dropout specification must be at the end of the string');
			dropout=false;
			break;
			
		case {'cust', 'custom'}
			assert(isempty(hasCustomFlag), 'multiple custom flag specification!');
			hasCustomFlag=true;
		
		otherwise
			if ~any(strcmp(supportedList,str2check))
				warning('Unexpected token %s found while parsing stereotype str=%s. It was ignored', tok, str);
			end
	end
end


end


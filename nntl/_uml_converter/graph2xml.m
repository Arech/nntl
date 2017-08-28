function [docNode] = graph2xml( gr,fname )
%GRAPH2XML Summary of this function goes here
%   Detailed explanation goes here

tli = find([gr(2:end).parentIid]==0);
assert(~isempty(tli));
assert(length(tli)==1,'Only one non-source data top level element permitted!');
tli = tli+1;

docNode=com.mathworks.xml.XMLUtils.createDocument('class');
docRoot = docNode.getDocumentElement();

pi = docNode.createProcessingInstruction('xml-stylesheet','type="text/xsl" href="layers2cpp.xsl"');
docNode.insertBefore(pi,docRoot);

docRoot.setAttribute('name', gr(tli).name);
assert( ~any( strcmp( 'Final', {gr(:).name} ) ) );
gr(tli).name='Final';

add_sourceNodes(docNode, gr);

add_layer_types(docNode, gr);

add_loss_addendums(docNode, gr)

layersNode = docNode.createElement('layers');
add_element(docNode, layersNode, gr, tli);
docRoot.appendChild(layersNode);

xmlwrite(fname, docNode);
end
%%
function add_layer_types(docNode, gr)
% we need this separate custom_types description, as each custom data type must be defined only once
% and it's rather hard to perform unique(set_of_values) during xslt. So it's better to add some redundancy to
% make life easier.
% By the same reason calculating unique combinations of layer+loss_addendum here

Lst = make_unique_layer_plus_modifier_list(gr);
if ~isempty(Lst)
	%LT=static_layer_types();
	rootNode = docNode.createElement('types');
	for ii=1:length(Lst)
		elm = docNode.createElement(layer_type2string(Lst(ii).type));
		%if Lst(ii).type==LT.custom
			%assert(~isempty(Lst(ii).customType));
		if ~isempty(Lst(ii).customType)
			elm.setAttribute('type', Lst(ii).customType);
		end
		if ~isempty(Lst(ii).modifier)
			elm.setAttribute('modifier', Lst(ii).modifier);
		end		
		rootNode.appendChild(elm);
	end	
	docNode.getDocumentElement().appendChild(rootNode);
end
end

function [Lst] = make_unique_layer_plus_modifier_list(gr)
grCnt = length(gr);

StrDescr=cell(grCnt,1);
Types=cell(grCnt,1);
CTypes = cell(grCnt,1);
Modifs = cell(grCnt,1);
for ii=2:grCnt
	[StrDescr{ii}, Types{ii}, CTypes{ii}, Modifs{ii}] = make_layer_plus_modifier(gr, ii);
end

nes = cellfun(@(c)~isempty(c),StrDescr);
StrDescr=StrDescr(nes);
Types=Types(nes);
CTypes = CTypes(nes);
Modifs = Modifs(nes);

[~, ia] = unique(StrDescr);
Lst = struct('type', Types(ia), 'customType', CTypes(ia), 'modifier',Modifs(ia));
end

function [strDescr, typ, ctyp, modif] = make_layer_plus_modifier(gr, idx)
LT=static_layer_types();
bLAEmpty = isempty(gr(idx).lossAdds);
assert(bLAEmpty || iscell(gr(idx).lossAdds));
assert(ischar(gr(idx).customType) || isempty(gr(idx).customType), ['Unexpected type of gr(' num2str(idx) ').customType']);
typ=gr(idx).type;
ctyp=[];
modif=[];
switch(typ)		
	case {LT.lfc, LT.lph, LT.lpv, LT.lid, LT.custom}
		ctyp = gr(idx).customType;
		assert(typ~=LT.custom || (~isempty(ctyp) && ischar(ctyp)), ['empty gr(' num2str(idx) ').customType']);
		if typ==LT.custom
			strDescr = ctyp;
		else
			strDescr = layer_type2string(typ);
			if ~isempty(ctyp)
				strDescr=[strDescr ctyp];
			end
		end
		if ~bLAEmpty
			modif = strjoin( sort(gr(idx).lossAdds),'_');
			strDescr=[strDescr modif];
		end
		dout = gr(idx).dropout;
		if ~isempty(dout)
			assert(ischar(dout));
			if isempty(modif)
				modif = dout;
			else
				modif = [modif '_' dout];
			end
			strDescr = [strDescr dout];
		end
		
	case LT.src
		%just skipping
		strDescr=[];
		
	case LT.unk
		error('There must be no layer of type LT.unk!');
	
	otherwise
		error(['Unexpected gr(' num2str(idx) ').type value = ' num2str(gr(idx).type)]);
end

end


%%
function add_loss_addendums(docNode, gr)
%rationale is the same as for the add_layer_types.

lossAdds = {gr(:).lossAdds};
nonempty=cellfun(@(c)~isempty(c),lossAdds);
lossAdds = lossAdds(nonempty);
if isempty(lossAdds), return; end
ulossAdds = unique([lossAdds{:}]);
if isempty(ulossAdds), return; end

rootNode = docNode.createElement('loss_addendums');
for ii=1:length(ulossAdds)
	curLa = ulossAdds{ii};
	laNode = docNode.createElement(curLa);
	%scanning throught the graph to find IDs of nodes which uses this LA
	for jj=1:length(nonempty)
		if nonempty(jj) && any(strcmp(curLa, gr(jj).lossAdds))
			idNode = docNode.createElement('id');
			idNode.setAttribute('ref', num2str(gr(jj).iid,'%d'));
			laNode.appendChild(idNode);
		end
	end	
	rootNode.appendChild(laNode);
end
docNode.getDocumentElement().appendChild(rootNode);

end

%%
function add_sourceNodes(docNode, gr)
LT=static_layer_types();
assert(strcmp(gr(1).name,'Data'));

rootNode = docNode.createElement('data');

if LT.src==gr(1).type
	add_element(docNode, rootNode, gr,1);
else
	chList = gr(1).childList;
	assert( all( arrayfun(@(c)~isempty(gr(c).nc),chList) ) );
	if all( arrayfun(@(c)isnumeric(gr(c).nc), chList ))
		rootNode.setAttribute('nc_num_only','');
	end
	
	for ii=1:length(chList)
		assert(isempty( gr(chList(ii)).childList ));
		add_element(docNode, rootNode, gr, chList(ii));
	end	
end

docNode.getDocumentElement().appendChild(rootNode);

end

%%
function add_element(docNode, rootNode, gr, iid)
LT=static_layer_types();
ltype = gr(iid).type;

bIsLPH = ltype == LT.lph;
bIsCompound = isLayerTypeCompound(ltype);

elmNode = docNode.createElement(layer_type2string(ltype));%'layer');
assert(~isempty(elmNode));

assert(~isempty(gr(iid).name));
elmNode.setAttribute('name', gr(iid).name);

if ~isempty(gr(iid).dropout)
	assert(ischar(gr(iid).dropout));
	elmNode.setAttribute('dout', gr(iid).dropout);
end

if ~isempty(gr(iid).customType)
	elmNode.setAttribute('type', gr(iid).customType);
end

if ~bIsCompound
	switch(ltype)
		case LT.lid
			assert(~isempty(gr(iid).innerIn) && isempty(gr(iid).nc));
			make_id_nodes_list(docNode, elmNode, 'nc', gr(iid).innerIn);
			
		case LT.src
			assert(~isempty(gr(iid).nc) && ((isnumeric(gr(iid).nc) && gr(iid).nc>0) || ischar(gr(iid).nc) ));
			if ischar(gr(iid).nc)
				elmNode.setAttribute('nc', gr(iid).nc);
			else
				elmNode.setAttribute('nc', num2str(gr(iid).nc,'%d'));
			end
			
		case LT.custom
			assert(~isempty(gr(iid).customType));
			%%elmNode.setAttribute('type', gr(iid).customType);
			
			if isempty(gr(iid).constr)
				assert(~isempty(gr(iid).nc));
				if isnumeric(gr(iid).nc)
					assert(gr(iid).nc>0);
					elmNode.setAttribute('nc', num2str(gr(iid).nc,'%d'));
				elseif ischar(gr(iid).nc)
					elmNode.setAttribute('nc', gr(iid).nc);
				else
					error('Unexpected .nc type');
				end
			else
				assert(isempty(gr(iid).nc));
				if ischar(gr(iid).constr)
					constr=gr(iid).constr;
				else
					constr='';
				end
				elmNode.setAttribute('constr', constr);
			end
			
		otherwise
			assert(~isempty(gr(iid).nc));
			if isnumeric(gr(iid).nc)
				assert(gr(iid).nc>0);
				elmNode.setAttribute('nc', num2str(gr(iid).nc,'%d'));
			elseif ischar(gr(iid).nc)
				elmNode.setAttribute('nc', gr(iid).nc);
			else
				error('Unexpected .nc type');
			end
	end
end

elmNode.setAttribute('id', num2str(iid,'%d'));

chList = gr(iid).childList;
assert(bIsCompound || isempty(chList));

for ii=1:length(chList)
	if bIsLPH
		phlEl = docNode.createElement('phl');
		assert(~isempty(phlEl));
		
		%phlEl.setAttribute('ofs', num2str(lph_phl(ii,1),'%d'));
		%phlEl.setAttribute('cnt', num2str(lph_phl(ii,2),'%d'));
		make_id_nodes_list(docNode, phlEl, 'ofs', gr(iid).phl_ref{ii,1} );
		make_id_nodes_list(docNode, phlEl, 'cnt', gr(iid).phl_ref{ii,2} );
		
		%layEl = docNode.createElement('layer');
		
		add_element(docNode, phlEl, gr, chList(ii));		
		%phlEl.appendChild(layEl);
		elmNode.appendChild(phlEl);
	else
		add_element(docNode, elmNode, gr, chList(ii));	
	end
end
	
rootNode.appendChild(elmNode);

end
%%
function make_id_nodes_list(docNode, apndTo, rootNodeName, iidList)
assert(~isempty(iidList));
if any(iidList > 0)
	nl = docNode.createElement(rootNodeName);
	for ii=1:length(iidList)
		if iidList(ii)>0
			idNode = docNode.createElement('id');
			idNode.setAttribute('ref', num2str(iidList(ii),'%d') );
			nl.appendChild(idNode);
		end
	end
	apndTo.appendChild(nl);
end
end
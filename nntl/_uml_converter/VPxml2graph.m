function [ gr ] = VPxml2graph( fname, bMoveDropoutToLph, bDropoutDefaultOn, dropoutDefaultType )
%VPXML2GRAPH Reads xml file with VisualParadigm's exported class diagram and converts it to neural network
%	architecture graph.
% Params:
%	- bMoveDropoutToLph - removes dropout from every uppermost children layer of LPH and sets dropout
%		for the whole LPH (optimal when every top-level children should have a dropout)
%	- bDropoutDefaultOn - determines how to treat a class when no dropout-related property specifed for
%		non-compound layers.
%	- bDropoutDefaultType - set's default type suffix for a dropout when it is absent
%
% note that LPT layers are not supported yet. Need to make a support for storing various parameters in NOTE
% shapes. Then it would be easy to specify any parameter for any layer (such as tiling count, loss addendums,
% layer's repeats and etc) via linked NOTE with parsable text. That's a TODO for future

if ~exist('bMoveDropoutToLph','var') || ~isscalar(bMoveDropoutToLph) || ~(islogical(bMoveDropoutToLph) || isnumeric(bMoveDropoutToLph))
	bMoveDropoutToLph=true;
end
bMoveDropoutToLph=logical(bMoveDropoutToLph);

if ~exist('bDropoutDefaultOn','var') || ~isscalar(bDropoutDefaultOn) || ~(islogical(bDropoutDefaultOn) || isnumeric(bDropoutDefaultOn))
	bDropoutDefaultOn=true;
end
bDropoutDefaultOn=logical(bDropoutDefaultOn);

if ~exist('dropoutDefaultType','var')
	if bDropoutDefaultOn
		dropoutDefaultType='DO';
	else
		dropoutDefaultType=[];
	end
end
assert(ischar(dropoutDefaultType) || isempty(dropoutDefaultType));

%% TODO: dropout handling is incostistent now - even if you turn dropout off with nodropout, the effective neurons count is influenced
%by diagram-global dropout rate. Need some special handling of such cases

gr = repairGraph( readShapes(xmlread(fname),bDropoutDefaultOn),bMoveDropoutToLph,dropoutDefaultType );
%After readShapes() .dropout property might have the following value:
%[] - empty - no dropout
%1 - dropout type by default
%string - property read from xml

end

%%
function cs=readShape(xtree, sNode, bIsPackage,bDropoutDefaultOn)
assert(sNode.hasAttributes());
atrs=sNode.getAttributes();
assert(~isempty(atrs));

%LT=static_layer_types();

%мы будем св€зывать данные по »ƒ модели (он универсальнее), но в диаграмме данные св€зываютс€ по
%»ƒ шейпа
iid = atrs.getNamedItem('model');
sid = atrs.getNamedItem('id');
name = atrs.getNamedItem('name');
yVal = atrs.getNamedItem('y');%по y-координате определ€ем направление св€зи (снизу вверх)
xVal = atrs.getNamedItem('x');%по x-координате определ€ем очерЄдность в LPH

if isempty(iid) || isempty(sid) || isempty(name) || isempty(yVal) || isempty(xVal)
	error('A Shape lacks mandatory properties');
end
iid = char(iid.getValue());
sid = char(sid.getValue());
name = char(name.getValue());
yVal = char(yVal.getValue());
xVal = char(xVal.getValue());
y=str2double(yVal);
x=str2double(xVal);
if isempty(iid) || isempty(sid) || isempty(name) ...
		|| isempty(yVal) || isnan(y) || y<0 || isempty(xVal) || isnan(x) || x<0
	error('Invalid Shape properties value');
end

assert(~strcmp('Final', name), 'Don`t call any shape `Final` - it`s a reserved keyword to designate the topmost layer');

if bIsPackage
	nc = [];
	[type, lossAdds, dout, isCustom] = readPackageType(xtree,iid);
	
	if ~isempty(isCustom) && isCustom
		custType = name;
	else
		custType=[];%we're expecting it to have one of the standard types (LPH or LPV)
		%i.e. a package can't have a custom type (design it as a separate feature detector and use as a
		%simple custom layer on this diagram.
	end
	
	constr = [];
	repeat = [];
else
	[nc,type,custType, constr, lossAdds, dout, repeat] = readClassModelProps(xtree,iid, name);
end

if isempty(dout) && ~ischar(dout)
	if bDropoutDefaultOn
		dout = 1;
	end
end
if islogical(dout)
	if dout
		dout = 1;
	else
		dout = [];
	end	
end

if ~isempty(lossAdds)
	lossAdds={lossAdds};
end

cs = struct('iid',iid,'sid',sid,'name',name,'y',y, 'x',x, 'nc', nc...
	, 'parentIid',[],'childList',0,'type', type, 'customType', custType ...
	, 'constr', constr, 'lossAdds', lossAdds, 'dropout', dout, 'repeat', repeat);
cs.childList={};
%parentIid and childList should be filled by a caller
end

%%
function [nc, type, custType, constr, lossAdds, dout, repeat]=readClassModelProps(xtree,modelId, name)
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();
LT=static_layer_types();

nc=[];
type=LT.lfc;
custType=[];
constr=[];
lossAdds=[];
dout=[];
repeat=[];

XP_expr = XP.compile( ['/Project/Models//Model[@modelType="Class" and @id="' modelId '"]']);
modelNode = XP_expr.evaluate(xtree, XPathConstants.NODESET);
assert(~isempty(modelNode) && modelNode.getLength()==1, ['None or multiple class modelId="' modelId '" found. Name=' name]);
modelNode = modelNode.item(0);

%read a nc attribute
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="nc"]/ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
v = XP_expr.evaluate(modelNode, XPathConstants.STRING);
if ~isempty(v)
	nc=str2double(v);
	if isnan(nc)
		fprintf(1, '**Beware, using string value `%s` for .nc value of model %s name %s. You take care for correctness!\n', v, modelId, name);
		nc = v;
	else
		assert(nc>0, ['Invalid neurons count value nc="' nc '" read from class modelId="' modelId '", Name=' name] );
	end
end

%read la (loss_addendum) attribute
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="la"]/ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
v = XP_expr.evaluate(modelNode, XPathConstants.STRING);
if isempty(v)
	XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="la"]/ModelProperties/TextModelProperty[@name="type"]/StringValue[@value]/@value');
	v = XP_expr.evaluate(modelNode, XPathConstants.STRING);
end
if ~isempty(v)
	%trying to extract loss addendum
	lossAdds = string2lossAddendum(v);
end

%read dropout attribute
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="dropout"]');
atrNode = XP_expr.evaluate(modelNode, XPathConstants.NODESET);
if ~isempty(atrNode) && atrNode.getLength()>0
	assert(atrNode.getLength()==1, ['Unexpected count of `dropout` attribute nodes! Name=' name]);
	dout=true;%it exists (empty value ([]) means it doesn't exist at all and default value should be used)

	XP_expr = XP.compile('ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
	v = lower(XP_expr.evaluate(atrNode.item(0), XPathConstants.STRING));
	if ~isempty(v)
		if any(strcmp(v,{'false','0','off'}))
			dout = false;
		else
			if ~any(strcmp(v,{'true','1','on'}))
				dout = upper(v);
			end
		end
	end
end
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="nodropout"]');
atrNode = XP_expr.evaluate(modelNode, XPathConstants.NODESET);
if ~isempty(atrNode) && atrNode.getLength()>0
	assert(atrNode.getLength()==1,'Unexpected count of `no_dropout` attribute nodes!');
	assert(isempty(dout) || (islogical(dout) && ~dout),['Dropout property has already been set for modelId="' modelId '"']);
	dout = false;
end

%read a type attribute

%note that the following XPath is slightly wrong (due to my old mistake). It queries TYPE of attribute, that is
%designated by a colon in attribute description, while actually we need (should use) a VALUE, that is written
%after equality sign
%XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="type"]/ModelProperties/TextModelProperty[@name="type"]/StringValue[@value]/@value');
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="type"]/ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
v = XP_expr.evaluate(modelNode, XPathConstants.STRING);
if ~isempty(v)
	type = string2layer_type(v);
	assert(~isLayerTypeCompound(type), ['Invalid(compound) layer type type="' v '" read from class modelId="' modelId '", Name=' name] );
	if type==LT.custom
		custType=v;
		
		% now we must check from 'constr' attribute existence and value
		XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="constr"]');
		atrNode = XP_expr.evaluate(modelNode, XPathConstants.NODESET);
		if ~isempty(atrNode) && atrNode.getLength()>0
			assert(atrNode.getLength()==1,'Unexpected count of `constr` attribute nodes!');
			constr=0;%at least, it exists (empty value ([]) means it doesn't exist at all)
			
			% .nc for custom constructed layers are taken from the layer type::nc_Final
			assert(isempty(nc), ['No .nc must be defined for custom constructed layer such as name=' name])
			%nc=[];%cleaning .nc value - we don't need it. It'll be read from customType
						
			XP_expr = XP.compile('ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
			v = XP_expr.evaluate(atrNode.item(0), XPathConstants.STRING);
			if ~isempty(v)
				constr = v;
			end
		end
	end
end

%read a repeat attribute
XP_expr = XP.compile('ChildModels/Model[@modelType="Attribute" and @name="repeat"]/ModelProperties/TextModelProperty[@name="initialValue"]/StringValue[@value]/@value');
v = XP_expr.evaluate(modelNode, XPathConstants.STRING);
if ~isempty(v)
	repeat=str2double(v);
	if isnan(repeat)
		fprintf(1, '**Beware, using string value `%s` for .repeat value of model %s name %s. You take care for correctness!\n', v, modelId, name);
		repeat = v;
	else
		assert(repeat>0, ['Invalid repeat value ="' repeat '" read from class modelId="' modelId '", Name=' name] );
	end
end

end
%%
function [pt, lossAdds, dout, isCust]=readPackageType(xtree,modelId)
%package type is specified via a stereotypes property.
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();

LT=static_layer_types();

pt=0;
lossAdds=[];
dout=[];
isCust=[];
bTried=false;

XP_expr = XP.compile( ['/Project/Models//Model[@modelType="Package" and @id="' modelId ...
	'"]/ModelProperties/ModelRefsProperty[@name="stereotypes"]/ModelRef/@id']);
stIidsSet = XP_expr.evaluate(xtree, XPathConstants.NODESET);
ss = stIidsSet.getLength();
if ss>0
	for ii=1:ss
		stIid = char(stIidsSet.item(ii-1).getValue());
		if ~isempty(stIid)
			bTried=true;
			XP_expr = XP.compile( ['/Project/Models//Model[@modelType="Stereotype" and @id="' stIid '"]/@name']);
			stName = XP_expr.evaluate(xtree, XPathConstants.STRING);
			assert(~isempty(stName),['Package stereotype "' stIid '" name is empty!']);
			
			[ tCand, lossAddsCand, dCand, isCustCand ] = parseStereotype(stName);
			
			assert(tCand>0);%by the string2layer_type nature it can't be 0
			if LT.custom==tCand
				%it could only specify a loss addendum or a dropout			
			else
				assert(isLayerTypeCompound(tCand),['Package model="' modelId '" has invalid type specification=' stName]);
				assert(0==pt || pt==tCand,['Uncoherent type specification found for package model="' modelId '". Last type specification=' stName]);
				pt = tCand;
			end

			if isempty(dout)
				dout = dCand;
			else
				assert(isempty(dCand) ...
					|| ((islogical(dout) && islogical(dCand) && dout==dCand)...
					|| (ischar(dout) && ischar(dCand) && strcmp(dout,dCand)))...
					,['Uncoherent dropout specification found for package model="' modelId '". Last type specification=' stName]);
			end

			%trying to extract loss addendum
			if ~isempty(lossAddsCand)
				assert(iscell(lossAddsCand));
				if isempty(lossAdds)
					lossAdds = lossAddsCand;
				else
					lossAdds = [lossAdds lossAddsCand{:}];
				end
			end
			
			if ~isempty(isCustCand)
				assert(islogical(isCustCand));
				assert(isempty(isCust),['Double custom flag specification found for model=' modelId]);
				isCust = isCustCand;
			end
		end
	end
	assert(~bTried || pt>0,['Type specification for package model="' modelId '" not found!']);
	
	%checking dropout setting
	if pt==LT.lpv %must be off
		assert(isempty(dout),['Apply dropout to a topmost layer of the LPV with model="' modelId '"']);
		dout=false;
	end
	
	if ~isempty(lossAdds)
		assert(iscell(lossAdds));
		assert(numel(lossAdds) == numel( unique(lossAdds) ), ['Non unique loss addendums specification found for model=' modelId]);
	end
end
end

%%
function shapes=readWholePackage(xtree,packageNode, bIsSrc,bDropoutDefaultOn)
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();

LT=static_layer_types();
shapes = def_shapeStruct();

%now lets read the package props first
shapes(1)=readShape(xtree, packageNode, true,bDropoutDefaultOn);
shapesCnt=1;
if bIsSrc
	%setting/checking the source package to have LPT type
	if shapes(shapesCnt).type==0, shapes(shapesCnt).type=LT.lph; end
	assert(shapes(shapesCnt).type == LT.lph, 'Source Data package type could be a LPH only');
end
packageIid=shapes(1).iid;

%reading child shapes
XP_child = XP.compile('ChildShapes/Shape');
chlList = XP_child.evaluate(packageNode, XPathConstants.NODESET);
assert(~isempty(chlList) && chlList.getLength()>0, 'There must be no empty packages!');

for ii=1:chlList.getLength()
	child = chlList.item(ii-1);
	assert(~isempty(child) && child.hasAttributes() && child.hasChildNodes());
	childType = char(child.getAttributes().getNamedItem('shapeType').getValue());
	
	assert(~isempty(childType));
	if ~any(strcmp(childType,{'Class','Package'}))
		fprintf(1,'*** skipping an unknown shape with type `%s`\n', childType);
		continue;
	end
	
	switch(childType)
		case 'Class'
			shp = readShape(xtree, child, false,bDropoutDefaultOn);
			shp.parentIid = packageIid;
			shapes(1).childList = [shapes(1).childList shp.iid];
			if bIsSrc
				shp.type = LT.src;
			end			
			shapesCnt=shapesCnt+1;
			shapes(shapesCnt)=shp;
			
		case 'Package'
			if bIsSrc
				error('There must be no subpackages inside the source data package');
			else
				shp = readWholePackage(xtree, child, false,bDropoutDefaultOn);
				assert(~isempty(shp));
				%updating parent node of the package node (it is always the first item)
				shp(1).parentIid = packageIid;
				shapes(1).childList = [shapes(1).childList shp(1).iid];
				
				shListCnt=length(shp);
				shapes(shapesCnt+1:shapesCnt+shListCnt) = shp;
				shapesCnt = shapesCnt+shListCnt;
			end
			
		otherwise
			error('WTF? Unexpected shapeType found in a package');		
	end
	
end

end

%%
function [shapes] = readShapes(xtree,bDropoutDefaultOn)
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();

LT=static_layer_types();
shapes = def_shapeStruct();
shapesCnt=1;%first shape for a source Data
bFoundSourceData=false;


XP_shapes = XP.compile('/Project/Diagrams/Diagram/Shapes/Shape');
topShapes = XP_shapes.evaluate(xtree, XPathConstants.NODESET);
assert(~isempty(topShapes) && topShapes.getLength()>1,'There must be at least two top level shapes on the diagram!');
for ii=1:topShapes.getLength()
	child = topShapes.item(ii-1);
	assert(~isempty(child) && child.hasAttributes() && child.hasChildNodes());
	atrs = child.getAttributes();
	
	childType = char(atrs.getNamedItem('shapeType').getValue());
	assert(~isempty(childType));
	if ~any(strcmp(childType,{'Class','Package'}))
		fprintf(1,'*** skipping an unknown shape with type `%s`\n', childType);
		continue;
	end
	
	childName = char(atrs.getNamedItem('name').getValue());
	assert(~isempty(childName));
	bIsSrc = strcmp('Data',childName);
	assert(~bFoundSourceData || ~bIsSrc, 'There must be no second Data shape in the diagram!');
	bFoundSourceData = bFoundSourceData || bIsSrc;
	
	switch(childType)
		case 'Class'
			if bIsSrc
				shpId = 1;
			else
				shapesCnt=shapesCnt+1;
				shpId = shapesCnt;
			end
			shapes(shpId) = readShape(xtree, child, false,bDropoutDefaultOn);
			if bIsSrc
				%error('Whats the point of using a single source node?');
				shapes(shpId).type = LT.src;
			end

		case 'Package'
			shList = readWholePackage(xtree, child, bIsSrc,bDropoutDefaultOn);
			shListCnt=length(shList);
			if bIsSrc
				assert(strcmp('Data',shList(1).name) && shListCnt>1);
								
				if shListCnt==2
					%dropping toplevel package and leaving only a single data source
					shapes(1) = shList(2);
					shapes(1).parentIid=[];
					shListCnt=0;
				else
					%saving package as a container for a set of data sources
					shapes(1) = shList(1);
					
					shListCnt=shListCnt-1;
					shapes(shapesCnt+1:shapesCnt+shListCnt) = shList(2:end);
				end
				
			else
				shapes(shapesCnt+1:shapesCnt+shListCnt) = shList;
			end
			shapesCnt = shapesCnt+shListCnt;

		otherwise
			error('WTF? Unexpected shapeType for a shape');
	end	
end
assert(bFoundSourceData,'There must be a single top level LPH package with the name "Data" to represent source data layers');

%assert(2==sum(cellfun(@(cv)isempty(cv), {shapes(:).parentIid})),'Only two top level package possible (and one of them must represent the source data)');

assert(shapesCnt == length(unique({shapes(:).iid})));
assert(shapesCnt == length(unique({shapes(:).sid})));
assert(shapesCnt == length(unique({shapes(:).name})), 'All shapes MUST have different names!');
assert(~any([shapes(:).type]==LT.unk),'There is at least one unknown/unset layer type');

% checking for .constr correct use
check_constr_use(shapes);

shapes=readConnections(shapes, xtree);

end
%%
function check_constr_use(gr)
otypes = {gr(:).customType};
nonemptyTypes=cellfun(@(c)~isempty(c),otypes);
otypes = otypes(nonemptyTypes);
types = unique(otypes);
if ~isempty(types)
	constrEmpty = {gr(:).constr};
	constrEmpty = cellfun(@(c)isempty(c), constrEmpty);
	constrEmpty = constrEmpty(nonemptyTypes);
	for ii=1:length(types)
		constrTypeValues = constrEmpty( strcmp(types{ii}, otypes ) );
		assert( all(constrTypeValues == constrTypeValues(1)), 'If a .constr property set for a custom type object, it must be set for every instance of that type' );
	end
end
end

%%
function shapes=readConnections(shapes, xtree)
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();

XP_conns = XP.compile('/Project/Diagrams/Diagram/Connectors/Connector[@shapeType="Association" or @shapeType="Dependency"]');
conns = XP_conns.evaluate(xtree, XPathConstants.NODESET);
connsCnt = conns.getLength();
assert(~isempty(conns) && connsCnt >0, 'Empty connectors list');

vl = cell(1,connsCnt);
connList = struct('sid1',vl,'sid2',vl,'nc',vl);
cCnt=0;

for ii=1:conns.getLength()
	cCnt=cCnt+1;
	connList(cCnt) = readConnection (xtree, conns.item(ii-1));	
end
%changing IDs of shapes and making connections
shapes=convertIds(shapes,connList);

end
%%
function shapes = convertIds(shapes, conns)
shapesCnt = length(shapes);

connSids1 = {conns(:).sid1};
connSids2 = {conns(:).sid2};
parentIids = {shapes(:).parentIid};
childList = {shapes(:).childList};

for ii=1:shapesCnt
	%мен€ем iid
	iid = shapes(ii).iid;
	shapes(ii).iid = ii;
	idxs = find(strcmp(iid,parentIids));
	if ~isempty(idxs)
		for jj=1:length(idxs), shapes(idxs(jj)).parentIid = ii; end
	end
	idxs = cellfun(@(cv)find(strcmp(iid,cv)), childList, 'UniformOutput', false);
	for jj=1:shapesCnt
		if ~isempty(idxs{jj})
			iidxs = idxs{jj};
			for kk=1:length(iidxs)
				shapes(jj).childList{iidxs(kk)} = ii;
			end			
		end
	end
	
	%по св€з€м: мен€ем sid. ќбновить nc пока нельз€, т.к. нужен второй числовой sid дл€ определени€
	%кому принадлежит соответствующий conns.nc
	sid = shapes(ii).sid;
	ci = find(strcmp(sid,connSids1));
	if ~isempty(ci)
		for jj=1:length(ci), conns( ci(jj) ).sid1=ii; end
	end
	
	ci = find(strcmp(sid,connSids2));
	if ~isempty(ci)
		for jj=1:length(ci), conns(ci(jj)).sid2=ii; end
	end
end

parentIids = {shapes(:).parentIid};
assert(all( cellfun(@(cv)isnumeric(cv), parentIids) ), ...
	'Some shapes doesnt have a corresponding parent shape on the diagram!');
idxs = find( cellfun(@(cv)isempty(cv), parentIids) );
for jj=1:length(idxs), shapes(idxs(jj)).parentIid = 0; end

%convert childList
for ii=1:shapesCnt
	if ~isempty(shapes(ii).childList)
		assert(all(cellfun(@(cv2)isnumeric(cv2), shapes(ii).childList) ), ...
			'Some parent shape doesnt have a corresponding child shape on the diagram!');		
	end
	shapes(ii).childList=cell2mat(shapes(ii).childList);
	assert(length(shapes(ii).childList) == length(unique(shapes(ii).childList)));
end
	
%assert(all( cellfun(@(cv)all(cellfun(@(cv2)isnumeric(cv2), cv)), {shapes(:).childList}) ), ...
%	'Some parent shape doesnt have a corresponding child shape on the diagram!');

assert(all( cellfun(@(cv)isnumeric(cv), {conns(:).sid1}) ) ...
	|| ~all( cellfun(@(cv)isnumeric(cv), {conns(:).sid2}) ),...
	'There are connections between unknown shapes!');

shapes = make_connections(rmfield(shapes,'sid'),conns);

end

%%
function shapes = make_connections(shapes,conns)
LT = static_layer_types();
CD = static_codegen_defs();

connCnt = length(conns);
%направленный граф будем описывать двум€ матрицами - исход€щих и вход€щих св€зей. Ќаправление у нас
%определ€етс€ значением координаты Y - св€зь всегда от большей Y к меньшей.
%shapesCnt = length(shapes);

shapes(1).in=[];
shapes(1).out=[];

	function placeEdge(fromI,toI, nc)
		assert(fromI~=toI,'Self connections are prohibited!');
		if nc>0
			if ~isempty(shapes(fromI).nc)
				if ischar(shapes(fromI).nc)
					error(['Shape `' shapes(fromI).name '` already have a .nc property specified as a string (`'...
						shapes(fromI).nc '`), while its outgoing connection has a numeric .nc(=' ...
						num2str(nc,'%d') ')!']);
				end
				assert(shapes(fromI).nc ==nc, ['Inconsistent neurons count for "' shapes(fromI).name ...
					'": it already has nc=' num2str(shapes(fromI).nc,'%d') ...
					' while a connection specifies nc=' num2str(nc,'%d')] );
			else
				shapes(fromI).nc=nc;
			end
		else
			% connection has no .nc specified. Checking if shape's attribute is also not specifed and if so
			% setting to a default value
			typ = shapes(fromI).type;
			if isempty(shapes(fromI).nc) && (typ == LT.lfc || typ == LT.src || (typ == LT.custom && isempty(shapes(fromI).constr)))
				assert(~isempty(shapes(fromI).name) && ischar(shapes(fromI).name));
				shapes(fromI).nc = [ CD.fullNcPfx shapes(fromI).name];
				disp(['* note: no suitable .nc value found for ' shapes(fromI).name ', using default ' shapes(fromI).nc]);
			end
		end
		assert(isempty(find(shapes(toI).in == fromI,1)), 'Hey! How did you managed to make two edges for the same two shapes?');
		assert(isempty(find(shapes(fromI).out == toI,1)), 'Hey! How did you managed to make two edges for the same two shapes?');
		shapes(toI).in = [shapes(toI).in fromI];
		shapes(fromI).out = [shapes(fromI).out toI];
	end

for ii=1:connCnt
	s1=conns(ii).sid1;
	s2=conns(ii).sid2;
	nc = conns(ii).nc;
	
	if shapes(s1).y > shapes(s2).y
		%edge from s1 to s2
		placeEdge(s1,s2,nc);		
	else
		%edge from s2 to s1
		placeEdge(s2,s1,nc);
	end	
end

%shapes=rmfield(shapes,'y');
end
%%
function c=readConnection(xtree, s)
assert(s.hasAttributes());
atrs=s.getAttributes();
assert(~isempty(atrs));

iid = atrs.getNamedItem('model');
sid1 = atrs.getNamedItem('from');
sid2 = atrs.getNamedItem('to');
assert( ~( isempty(iid) || isempty(sid1) || isempty(sid2)), 'A connector lacks mandatory properties');

iid = char(iid.getValue());
sid1 = char(sid1.getValue());
sid2 = char(sid2.getValue());
assert(~(isempty(iid) || isempty(sid1) || isempty(sid2)), 'Invalid connector properties value');

nc = readNcFromNameAttr(atrs);
%пытаемс€ прочитать число нейронов из свойств модели
ncModel = readNcFromConnModel(xtree, iid);
if ncModel>0
	if nc>0
		assert(nc==ncModel, 'Neurons count specified in connection name of the shape mismatches specified in the connection model');
	else
		nc=ncModel;
	end
end

c = struct('sid1', sid1,'sid2',sid2, 'nc',nc);
end
%%
function nc=readNcFromNameAttr(atrs)
nc = atrs.getNamedItem('name');
if ~isempty(nc)
	nc = char(nc.getValue());
	if ~isempty(nc)
		nc = str2double(nc);
		if isnan(nc) || nc<=0
			error('Connection name must be a digit specifying a count of neurons of the undelying layer');
		end
	end
end
if isempty(nc)
	nc = 0;
end
end


%%
function ncModel = readNcFromConnModel(xtree, modelId)
import javax.xml.xpath.*
XP = XPathFactory.newInstance().newXPath();

%XP_expr = XP.compile( ['/Project/Models/Model[@modelType="ModelRelationshipContainer"]/ChildModels//Model[@id="' modelId '"]']);
XP_expr = XP.compile( ['/Project//Model[@id="' modelId '"]']);
mdlNode = XP_expr.evaluate(xtree, XPathConstants.NODESET);
if isempty(mdlNode) || mdlNode.getLength()~=1 || ~mdlNode.item(0).hasAttributes()
	error('Connection model node not found or found too many of them!');
end
mdlNode = mdlNode.item(0);
%из атрибута name
ncModel=readNcFromNameAttr(mdlNode.getAttributes());

%из одного из концов
XP_expr = XP.compile('FromEnd/Model[@modelType="AssociationEnd" and @name]');
endNode = XP_expr.evaluate(mdlNode, XPathConstants.NODE);
if ~isempty(endNode)
	nc = readNcFromNameAttr(endNode.getAttributes());
	if ncModel>0
		if nc>0 && nc~=ncModel
			error('Connection name attribute mismatches the name of the FromEnd');
		end
	else
		ncModel = nc;
	end
end

XP_expr = XP.compile('ToEnd/Model[@modelType="AssociationEnd" and @name]');
endNode = XP_expr.evaluate(mdlNode, XPathConstants.NODE);
if ~isempty(endNode)
	nc = readNcFromNameAttr(endNode.getAttributes());
	if ncModel>0
		if nc>0 && nc~=ncModel
			error('Connection name attribute mismatches the name of the FromEnd');
		end
	else
		ncModel = nc;
	end
end
end

%%
function dShpList = def_shapeStruct()
vl=cell(1,1);
dShpList = struct('iid',vl,'sid',vl,'name',vl,'y',vl, 'x',vl, 'nc',vl,...
	'parentIid',vl,'childList',vl,'type',vl,'customType',vl,'constr',vl,'lossAdds',vl, 'dropout',vl, ...
	'repeat', vl);
%when layer is of custom type, and there is a 'constr' attribute set, it implies two things:
% 1. customType (type attribute) specifies a struct data type similar to obtained from our xslt-transform.
% 2. nc_ and .lFinal member values are obtained from this type. nc MUST NOT be specified in diagram
% 3. .constr attribute value if set must be a string with actual object construction arguments
%
% .repeat propery is allowed only for direct child of LPH. It's designed to pass all incoming neurons
%	to .repeat instances of the layer simultaneously
% Note that it seems that whole idea of .repeat propery is just a glitch, so it will probably be removed

end
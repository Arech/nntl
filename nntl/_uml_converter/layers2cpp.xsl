<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="text"/>
<xsl:variable name="generatedBaseVersion" select="1" />
<xsl:variable name="generatedBaseNPVersion" select="1" />
<!-- generatedBaseVersion defines a suffix of file name to #include "_xml_generated_base_.h" -->
<xsl:strip-space elements="*"/>

<!-- =================================================================== -->
<!-- Rendering starts here -->
<xsl:template match="/">#pragma once

//WARNING: This file was autogenerated. All changes will be lost should it be re-generated again!

#include &quot;_xml_generated_base_<xsl:value-of select="$generatedBaseVersion"/>.h&quot;
// See an example of suitable version of _xml_generated_base_<xsl:value-of select="$generatedBaseVersion"/>.h file below after the struct <xsl:value-of select="/class/@name"/> definition

template&lt;typename P&gt;
struct <xsl:value-of select="/class/@name"/> : public _xml_generated_base_<xsl:value-of select="$generatedBaseVersion"/>&lt;typename P::real_t, typename P::PL<xsl:value-of select="/class/@name"/>&gt; {	
	//////////////////////////////////////////////////////////////////////////
	// Types aliases/definitions
	typedef typename P::PLId PLId;
	<xsl:if test='/class/types'>
		<xsl:text>
	// custom or enhanced standard classes
	</xsl:text>
		<!-- aliasing/declaring types of custom layers and regular layers with various type additions -->
		<xsl:for-each select='/class/types/*'>
			<xsl:choose>
				<xsl:when test='name()="lph" or name()="lpv"'>
					<xsl:text>template&lt;typename TupleT&gt; using </xsl:text>
						<xsl:call-template name="makeLayerType"><xsl:with-param name='istypedef' select='1'/></xsl:call-template>
						<xsl:text> = typename PL::template </xsl:text>
						<xsl:call-template name="makeLayerType"><xsl:with-param name='istypedef' select='1'/></xsl:call-template><xsl:text>&lt;TupleT&gt;;
	</xsl:text>
				</xsl:when>
				<xsl:when test='name()="lid"'>
					<!-- skipping it entirely -->
				</xsl:when>
				<xsl:otherwise>
					<xsl:text>typedef typename PL::</xsl:text><xsl:call-template name="makeLayerType"><xsl:with-param name='istypedef' select='1'/></xsl:call-template>
						<xsl:text> </xsl:text><xsl:call-template name="makeLayerType"><xsl:with-param name='istypedef' select='1'/></xsl:call-template><xsl:text>;
	</xsl:text>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each>
	</xsl:if>
	
	<!-- aliasing/declaring types of layer's loss_addendums-->
	<xsl:if test='/class/loss_addendums'>
		<xsl:text>
	// loss_addendum's classes
	</xsl:text>
		<xsl:for-each select='/class/loss_addendums/*'>
			<!-- aliasing loss_addendum type to be used later -->
			<xsl:text>typedef typename PL::</xsl:text><xsl:call-template name="makeLossAddendumType"/><xsl:text> </xsl:text><xsl:call-template name="makeLossAddendumType"/><xsl:text>;
	</xsl:text>
		</xsl:for-each>
	</xsl:if>
	
	<xsl:call-template name="describeSrc"/>
	
	<xsl:text>
	//////////////////////////////////////////////////////////////////////////
	// layer objects declaration
	</xsl:text>
	<xsl:for-each select="/class/layers/*[@name]">
		<xsl:apply-templates select='.' mode='describe'/>
	</xsl:for-each>
		
	<xsl:text>//////////////////////////////////////////////////////////////////////////
	// Methods
	~</xsl:text><xsl:value-of select="/class/@name"/>() noexcept{}
	
	<xsl:value-of select="/class/@name"/>() noexcept 
		: <xsl:for-each select="/class/layers/*[@name]">
		<xsl:apply-templates select='.' mode='construct'><xsl:with-param name='isfirst' select='1'/></xsl:apply-templates>
	</xsl:for-each>
	{
		//setting dropout where applicable
		if (dropoutAlive &gt; real_t(0) &amp;&amp; dropoutAlive &lt; real_t(1))
			lFinal.for_each_layer(::nntl::hlpr_layer_set_dropoutPercentActive&lt;real_t&gt;(dropoutAlive));
		<xsl:call-template name="initLossAddendums"/>
	}
};

<xsl:call-template name="xmlGeneratedBaseSample"/>
</xsl:template>

<!-- =================================================================== -->
<!-- Generates a string of layer's type -->
<xsl:template name='makeLayerType'>
<xsl:param name='istypedef'>0</xsl:param>
<xsl:choose>
	<xsl:when test="name()='lfc'">
		<xsl:text>PLFC</xsl:text>
	</xsl:when>
	<xsl:when test="name()='lph'">
		<xsl:text>PLPH</xsl:text>
	</xsl:when>
	<xsl:when test="name()='lpv'">
		<xsl:text>PLPV</xsl:text>
	</xsl:when>
	<xsl:when test="name()='lid'">
		<xsl:text>PLId</xsl:text>
	</xsl:when>
	<xsl:when test="name()='custom'">
		<xsl:value-of select='@type'/>
	</xsl:when>
	<xsl:otherwise>
		<xsl:message terminate="yes">Unexpected layer name() was found</xsl:message>
	</xsl:otherwise>
</xsl:choose>

<xsl:choose>
	<xsl:when test="$istypedef=0">
		<xsl:call-template name="appendLossAddendumSfx"><xsl:with-param name='curId'><xsl:value-of select='@id'/></xsl:with-param></xsl:call-template>
		<xsl:if test='@dout'>
			<xsl:text>_</xsl:text><xsl:value-of select='@dout'/>
		</xsl:if>
	</xsl:when>
	<xsl:otherwise>
		<xsl:if test='@modifier'>
			<xsl:text>_</xsl:text><xsl:value-of select='@modifier'/>
		</xsl:if>
	</xsl:otherwise>
</xsl:choose>
</xsl:template>

<!-- =================================================================== -->
<!-- Applying initial loss_addendum scales -->
<xsl:template name='initLossAddendums'>

<xsl:for-each select='/class/layers//*[@name and @id]'>
	<xsl:variable name="curId"><xsl:value-of select='@id'/></xsl:variable>
	<xsl:variable name="curLayerName">l<xsl:value-of select='@name'/></xsl:variable>
	<xsl:for-each select='/class/loss_addendums//*[@ref=$curId]/..'>
		<xsl:text></xsl:text><xsl:value-of select='$curLayerName'/>.addendum&lt;<xsl:call-template name="makeLossAddendumType"
			/>&gt;().scale(PL::<xsl:value-of select='$curLayerName'/>_<xsl:value-of select='name()'/><xsl:text>Scale);
		</xsl:text>
	</xsl:for-each>
</xsl:for-each>

</xsl:template>

<!-- =================================================================== -->
<!-- constructing layer objects -->
<xsl:template match='lph' mode='construct'>
	<xsl:param name='isfirst'>0</xsl:param>
	
	<xsl:for-each select="./phl">
		<xsl:variable name="realpos"><xsl:value-of select='position()'/></xsl:variable>
		<xsl:for-each select="./*[@name]">
			<xsl:if test='$realpos=1 and position()=1'>
				<xsl:apply-templates select='.' mode='construct'><xsl:with-param name='isfirst' select='$isfirst'/></xsl:apply-templates>
			</xsl:if>
			<xsl:if test='not($realpos=1) or position() &gt; 1'>
				<xsl:apply-templates select='.' mode='construct'><xsl:with-param name='isfirst' select='0'/></xsl:apply-templates>
			</xsl:if>
		</xsl:for-each>
	</xsl:for-each>

	<xsl:text>
		, l</xsl:text><xsl:value-of select='@name'/>(<xsl:call-template name="makeLayerName"/><xsl:text>
			, ::std::make_tuple(</xsl:text>
	<xsl:for-each select="./phl">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'>,</xsl:with-param>
			<xsl:with-param name='breakline' select='1'/><xsl:with-param name='addspacing'><xsl:text>	</xsl:text></xsl:with-param>
		</xsl:call-template>
		<xsl:text>make_PHL(l</xsl:text><xsl:value-of select='./*[@name]/@name'/><xsl:text></xsl:text>
		<xsl:if test='./*[@name and @constr and name()="custom" and @type]'>
			<xsl:text>.lFinal</xsl:text>
		</xsl:if><xsl:text>, </xsl:text>
		
		<xsl:if test='./ofs'>
			<xsl:for-each select="./ofs/id">
				<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'> +</xsl:with-param>
					<xsl:with-param name='breakline' select='4'/><xsl:with-param name='addspacing'><xsl:text>		</xsl:text></xsl:with-param>
				</xsl:call-template>
				<xsl:text>nc_</xsl:text><xsl:call-template name="id2name"/>
			</xsl:for-each>
		</xsl:if>
		<xsl:if test='not(./ofs)'>
			<xsl:text>0</xsl:text>
		</xsl:if>
		
		<xsl:text>, </xsl:text>
		<xsl:for-each select="./cnt/id">
			<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'> +</xsl:with-param>
				<xsl:with-param name='breakline' select='4'/><xsl:with-param name='addspacing'><xsl:text>		</xsl:text></xsl:with-param>
			</xsl:call-template>
			<xsl:text>nc_</xsl:text><xsl:call-template name="id2name"/>
		</xsl:for-each>
		
		<xsl:text>)</xsl:text>
	</xsl:for-each>
	<xsl:text>
		))</xsl:text>
</xsl:template>

<xsl:template match='lpv' mode='construct'>
	<xsl:param name='isfirst'>0</xsl:param>
	
	<xsl:for-each select="./*[@name]">
		<xsl:if test='position()=1'>
			<xsl:apply-templates select='.' mode='construct'><xsl:with-param name='isfirst' select='$isfirst'/></xsl:apply-templates>
		</xsl:if>
		<xsl:if test='position() &gt; 1'>
			<xsl:apply-templates select='.' mode='construct'><xsl:with-param name='isfirst' select='0'/></xsl:apply-templates>
		</xsl:if>
	</xsl:for-each>
	
	<xsl:text>
		, l</xsl:text><xsl:value-of select='@name'/>(<xsl:call-template name="makeLayerName"/><xsl:text>, ::std::tie(</xsl:text>
	<xsl:for-each select="./*[@name]">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'>,</xsl:with-param><xsl:with-param name='breakline' select='5'/>
			<xsl:with-param name='addspacing'><xsl:text>	</xsl:text></xsl:with-param></xsl:call-template>
		<xsl:text>l</xsl:text><xsl:value-of select='@name'/><xsl:text></xsl:text>
		<xsl:if test='@constr and name()="custom" and @type'>
			<xsl:text>.lFinal</xsl:text>
		</xsl:if>
	</xsl:for-each>
	<xsl:text>))</xsl:text>
</xsl:template>

<xsl:template match='lid' mode='construct'>
	<xsl:param name='isfirst'>0</xsl:param>
	<xsl:call-template name="beautifyConstruct"><xsl:with-param name='isfirst' select='$isfirst'/></xsl:call-template>
	
	<xsl:text>l</xsl:text><xsl:value-of select='@name'/>(<xsl:call-template name="makeLayerName"/><xsl:text>)</xsl:text>
</xsl:template>

<!-- TODO: custom layer types could probably have their own initialization schemes, we have to deal with it somehow (xsl:import/xsl:include ?)-->
<xsl:template match='lfc|custom' mode='construct'>
	<xsl:param name='isfirst'>0</xsl:param>
	<xsl:call-template name="beautifyConstruct"><xsl:with-param name='isfirst' select='$isfirst'/></xsl:call-template>
	
	<xsl:text>l</xsl:text><xsl:value-of select='@name'/><xsl:text>(</xsl:text>
	<xsl:if test='name()="custom" and @constr'>
		<xsl:text></xsl:text><xsl:value-of select='@constr'/>
	</xsl:if>
	<xsl:if test='not(name()="custom" and @constr)'>
		<xsl:text></xsl:text><xsl:call-template name="makeLayerName"/>, nc_<xsl:value-of select='@name'/>, learningRate<xsl:text></xsl:text>
	</xsl:if>
	
	<xsl:text>)</xsl:text>
</xsl:template>

<xsl:template name='beautifyConstruct'>
	<xsl:param name='isfirst'>0</xsl:param>
	<xsl:if test='$isfirst=0'>
		<xsl:text>
		, </xsl:text>
	</xsl:if>
</xsl:template>
	
<!-- =================================================================== -->
<!-- declaring layer objects during "describe" phase -->
<xsl:template match='lph' mode='describe'>
	<!-- inner layers are described first -->
	<xsl:for-each select="./phl">
		<xsl:for-each select="./*[@name]">
			<xsl:apply-templates select='.' mode='describe'/>
		</xsl:for-each>
	</xsl:for-each>

	<!-- defining layer's neurons count as a sum of inner layer's neurons -->
	<xsl:text>
	static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/><xsl:text> = </xsl:text>
	<xsl:for-each select="./phl/*[@name]">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'> +</xsl:with-param>
			<xsl:with-param name='breakline' select='4'/></xsl:call-template>
		<xsl:text>nc_</xsl:text><xsl:value-of select='@name'/>
	</xsl:for-each>
	<xsl:text>;
	</xsl:text>
	
	<xsl:call-template name="makeLayerType"/><xsl:text>&lt;::std::tuple&lt;</xsl:text>	
	<!-- PHLs list -->
	<xsl:for-each select="./phl/*[@name]">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'>,</xsl:with-param>
			<xsl:with-param name='breakline' select='3'/></xsl:call-template>
		<xsl:text>::nntl::PHL&lt;decltype(l</xsl:text><xsl:value-of select='@name'/><xsl:text></xsl:text>
		<xsl:if test='@constr and name()="custom" and @type'>
			<xsl:text>.lFinal</xsl:text>
		</xsl:if>
		<xsl:text>)&gt;</xsl:text>
	</xsl:for-each>
	<xsl:text>&gt;&gt; l</xsl:text><xsl:value-of select='@name'/><xsl:text>;	
	
	</xsl:text>
</xsl:template>

<xsl:template match='lpv' mode='describe'>
	<xsl:for-each select="./*[@name]">
		<xsl:apply-templates select='.' mode='describe'/>
	</xsl:for-each>
	
	<xsl:text>static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/> = nc_<xsl:value-of select='./*[last()]/@name'/>;<xsl:text>
	</xsl:text>
	
	<xsl:call-template name="makeLayerType"/><xsl:text>&lt;::std::tuple&lt;</xsl:text>	
	<xsl:for-each select="./*[@name]">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'>,</xsl:with-param><xsl:with-param name='breakline' select='4'/></xsl:call-template>
		<xsl:text>decltype(l</xsl:text><xsl:value-of select='@name'/><xsl:text></xsl:text>
		<xsl:if test='@constr and name()="custom" and @type'>
			<xsl:text>.lFinal</xsl:text>
		</xsl:if>
		<xsl:text>)&amp;</xsl:text>
	</xsl:for-each>
	<xsl:text>&gt;&gt; l</xsl:text><xsl:value-of select='@name'/><xsl:text>;
	
	</xsl:text>
</xsl:template>

<xsl:template match='lid' mode='describe'>
	<xsl:text>static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/><xsl:text> = </xsl:text>
	<xsl:for-each select="./nc/id">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'> +</xsl:with-param><xsl:with-param name='breakline' select='3'/></xsl:call-template>
		<xsl:text>nc_</xsl:text><xsl:call-template name="id2name"/>
	</xsl:for-each>
	<xsl:text>;	
	</xsl:text><xsl:call-template name="makeLayerType"/><xsl:text> l</xsl:text><xsl:value-of select='@name'/><xsl:text>;	
	</xsl:text>
</xsl:template>

<xsl:template match='custom' mode='describe'>
	<xsl:text>static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/><xsl:text> = </xsl:text>
	<xsl:if test='@constr'>
		<xsl:text></xsl:text><xsl:value-of select='@type'/><xsl:text>::nc_Final;
	</xsl:text>
	</xsl:if>
	<xsl:if test='not(@constr)'>
		<xsl:if test='@nc=string(number(@nc))'>
			<xsl:text>_getNC(</xsl:text>
		</xsl:if>
		<xsl:if test='not(@nc=string(number(@nc)))'>
			<xsl:text>_getNCnoMul(</xsl:text>
		</xsl:if>
		<xsl:text></xsl:text><xsl:value-of select='@nc'/><xsl:text>);
	</xsl:text>
	</xsl:if>
	
	<!--<xsl:text></xsl:text><xsl:value-of select='@type'/><xsl:text> l</xsl:text><xsl:value-of select='@name'/><xsl:text>;	-->
	<xsl:call-template name="makeLayerType"/><xsl:text> l</xsl:text><xsl:value-of select='@name'/><xsl:text>;	
	</xsl:text>
</xsl:template>

<xsl:template match='lfc' mode='describe'>
	<xsl:text>static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/><xsl:text> = </xsl:text>
	<xsl:if test='@nc=string(number(@nc))'>
		<xsl:text>_getNC(</xsl:text>
	</xsl:if>
	<xsl:if test='not(@nc=string(number(@nc)))'>
		<xsl:text>_getNCnoMul(</xsl:text>
	</xsl:if>
	<xsl:text></xsl:text><xsl:value-of select='@nc'/><xsl:text>);
	</xsl:text>
	
	<xsl:call-template name="makeLayerType"/><xsl:text> l</xsl:text><xsl:value-of select='@name'/><xsl:text>;
	</xsl:text>
</xsl:template>

<xsl:template match='*' mode='describe'>
	<xsl:message terminate="yes">Unexpected layer tag <xsl:value-of select='name()'/> was found. @name=<xsl:value-of select='@name'/></xsl:message>
</xsl:template>

<!-- =================================================================== -->
<!-- this routine emits source data description -->
<xsl:template name='describeSrc'>
	<xsl:text>
	//////////////////////////////////////////////////////////////////////////
	// Source data description:
	</xsl:text>
	<xsl:if test='/class/data[@nc_num_only]'>
		<xsl:for-each select="/class/data/src">
			<xsl:variable name="curPos"><xsl:value-of select='position()'/></xsl:variable>
			<xsl:text>// (</xsl:text><xsl:value-of select='sum(../src[position() &lt; $curPos]/@nc)'/>:<xsl:value-of select='sum(../src[position() &lt;= $curPos]/@nc)'/>) <xsl:value-of select='@name'/> (nc=<xsl:value-of select='@nc'/><xsl:text>)
	</xsl:text>
		</xsl:for-each>
	</xsl:if>
	<xsl:if test='not(/class/data[@nc_num_only])'>
		<xsl:for-each select="/class/data/src">
			<xsl:text>// </xsl:text><xsl:value-of select='@name'/><xsl:text> (nc=</xsl:text><xsl:value-of select='@nc'/><xsl:text>)
	</xsl:text>
		</xsl:for-each>
	</xsl:if>
	
	<xsl:for-each select="/class/data/src">
		<xsl:text>static constexpr neurons_count_t nc_</xsl:text><xsl:value-of select='@name'/> = <xsl:value-of select='@nc'/>;<xsl:text>
	</xsl:text>
		<!-- <xsl:if test='@nc_string'> -->
		<xsl:if test='not(@nc=string(number(@nc)))'>
			<xsl:text>static_assert(0 &lt; nc_</xsl:text><xsl:value-of select='@name'/>, "Invalid .nc property value for <xsl:value-of select='@name'/><xsl:text>!");
	</xsl:text>
		</xsl:if>
	</xsl:for-each>
	
	<xsl:text>
	// whole receptive field length
	static constexpr neurons_count_t RFLen = </xsl:text>
	<xsl:for-each select="/class/data/src">
		<xsl:call-template name="beautifyEnumeration"><xsl:with-param name='separator'> +</xsl:with-param>
			<xsl:with-param name='breakline' select='5'/></xsl:call-template>
		<xsl:text>nc_</xsl:text><xsl:value-of select='@name'/>
	</xsl:for-each>
	
	<xsl:text>;</xsl:text>
</xsl:template>


<!-- =================================================================== -->
<!-- various helpers -->
<xsl:template name='makeLossAddendumType'>
	<xsl:text>LA_</xsl:text><xsl:value-of select="name()"/><xsl:text>_t</xsl:text>
</xsl:template>

<xsl:template name='appendLossAddendumSfx'>
	<xsl:param name='curId'>-1</xsl:param>
	<xsl:if test='/class/loss_addendums//*[@ref=$curId]'>
		<!-- enumerating suitable loss_addendums -->
		<xsl:for-each select="/class/loss_addendums//*[@ref=$curId]/..">
			<xsl:text>_</xsl:text><xsl:value-of select="name()"/>
		</xsl:for-each>
	</xsl:if>
</xsl:template>

<xsl:template name='makeLayerName'>
	<xsl:text>"</xsl:text><xsl:value-of select="/class/@name"/>__l<xsl:value-of select='@name'/><xsl:text>"</xsl:text>
</xsl:template>

<xsl:template name='beautifyEnumeration'>
	<xsl:param name='separator'>,</xsl:param>
	<xsl:param name='breakline'>3</xsl:param>
	<xsl:param name='addspacing'></xsl:param>
	<xsl:if test='position() &gt; 1 and position() mod $breakline = 0'>
		<xsl:text>
		</xsl:text><xsl:value-of select='$addspacing'/>
	</xsl:if>
	<xsl:if test='position() &gt; 1'>
		<xsl:text></xsl:text><xsl:value-of select='$separator'/><xsl:text> </xsl:text>
	</xsl:if>
</xsl:template>

<xsl:template name='id2name'>
	<xsl:variable name="iid"><xsl:value-of select='@ref'/></xsl:variable>
	<xsl:text></xsl:text><xsl:value-of select='/*//*[@id=$iid]/@name'/>
</xsl:template>



<!-- =================================================================== -->
<!-- =================================================================== -->
<!-- base classes description -->
<!-- make a routing for saving/loading weights. BTW, it should be independent from the layer indexing -->
<xsl:template name='xmlGeneratedBaseSample'>
<xsl:text>
//////////////////////////////////////////////////////////////////////////
// Example of suitable version of _xml_generated_base_NP_</xsl:text><xsl:value-of select="$generatedBaseNPVersion"/><xsl:text>.h:
//////////////////////////////////////////////////////////////////////////
/*
#pragma once
struct _xml_generated_base_NP_</xsl:text><xsl:value-of select="$generatedBaseNPVersion"/><xsl:text> {
	//////////////////////////////////////////////////////////////////////////
	// routines for saving/loading weights should be used only after assembling layers in a stack.
protected:
	template&lt;typename _L, class ArchiveT&gt;
	static ::std::enable_if_t&lt;::nntl::layer_has_gradworks&lt;_L&gt;::value&gt; _saveWeights(_L&amp; l, ArchiveT&amp; ar, typename ArchiveT::ErrorCode&amp; er)noexcept {
		if (er == ArchiveT::ErrorCode::Success){
			NNTL_ASSERT(ArchiveT::ErrorCode::Success == ar.get_last_error());
			auto&amp; W = l.get_weights();
			ar &lt;&lt; ::nntl::serialization::make_nvp(l.get_custom_name(), W);
			er = ar.get_last_error();
			NNTL_ASSERT(ArchiveT::ErrorCode::Success == er);
		}
	}
	
	template&lt;typename _L, class ArchiveT&gt;
	static ::std::enable_if_t&lt;!::nntl::layer_has_gradworks&lt;_L&gt;::value&gt; _saveWeights(_L&amp; l, ArchiveT&amp; ar, typename ArchiveT::ErrorCode&amp; er)noexcept {}	
	
	//////////////////////////////////////////////////////////////////////////
	
	struct _DummyOp{
		template&lt;typename _L&gt; void operator()(_L&amp; l, const bool bWeightsHasBeenRead)noexcept{}
	};
	
	template&lt;typename _L, class ArchiveT, typename OpT&gt;
	static ::std::enable_if_t&lt;::nntl::layer_has_gradworks&lt;_L&gt;::value&gt; _loadWeights(_L&amp; l, ArchiveT&amp; ar, typename ArchiveT::ErrorCode&amp; er
		, const unsigned int instanceId, OpT&amp;&amp; op = _DummyOp())noexcept
	{
		if (er == ArchiveT::ErrorCode::Success){
			NNTL_ASSERT(ArchiveT::ErrorCode::Success == ar.get_last_error());
			
			typename _L::realmtx_t W;
			char _buf[_L::layerNameMaxChars];
			const char* pName;

			if (instanceId){
				snprintf(_buf, sizeof(_buf), "%s__%d", l.get_custom_name(), instanceId);
				pName = _buf;
			}else pName = l.get_custom_name();
			
			ar &gt;&gt; ::nntl::serialization::make_nvp(pName, W);
			er = ar.get_last_error();
			if (ArchiveT::ErrorCode::Success == er){
				//if (instanceId) {
					//::std::cout &lt;&lt; "reading " &lt;&lt; pName &lt;&lt; " into " &lt;&lt; l.get_layer_name_str() &lt;&lt; ::std::endl;
					//::std::forward&lt;OpT&gt;(op)(l);
				//}
				const bool bRead = l.set_weights(::std::move(W));
				if (!bRead) er = ArchiveT::ErrorCode::FailedToAssignDestinationVar;
				::std::forward&lt;OpT&gt;(op)(l, bRead);
			}else{
				if (instanceId){
					//it's ok to found no weights for some layers when reading by instance number
					ar._drop_last_error();
					er = ArchiveT::ErrorCode::Success;
				}else NNTL_ASSERT(!"Failed to read weight matrix");
				::std::forward&lt;OpT&gt;(op)(l, false);
			}
		}
	}
	
	template&lt;typename _L, class ArchiveT, typename OpT&gt;
	static ::std::enable_if_t&lt;!::nntl::layer_has_gradworks&lt;_L&gt;::value&gt; _loadWeights(_L&amp; l, ArchiveT&amp; ar, typename ArchiveT::ErrorCode&amp; er
		, const unsigned int instanceId, OpT&amp;&amp; op = _DummyOp())noexcept {}
	
public:
	
	template&lt;class ArchiveT&gt;
	typename ArchiveT::ErrorCode saveWeights(ArchiveT&amp; ar)noexcept{
		auto er = ar.get_last_error();
		NNTL_ASSERT(ArchiveT::ErrorCode::Success == er);
		
		lFinal.for_each_layer_down([&amp;ar, &amp;er](auto&amp; l)noexcept{
			_saveWeights(l, ar, er);
		});		
		return er;
	}
	
	template&lt;class ArchiveT, typename OpT&gt;
	typename ArchiveT::ErrorCode loadWeights(ArchiveT&amp; ar, const unsigned int instanceId = 0, OpT&amp;&amp; op = _DummyOp())noexcept{
		auto er = ar.get_last_error();
		
		NNTL_ASSERT(ArchiveT::ErrorCode::Success == er);		
		if (ArchiveT::ErrorCode::Success == er){
			lFinal.for_each_layer_down([&amp;ar, &amp;er, instanceId, &amp;F{ ::std::forward&lt;OpT&gt;(op) }](auto&amp; l)noexcept{
				_loadWeights(l, ar, er, instanceId, ::std::forward&lt;OpT&gt;(F));
			});
		}
		return er;
	}
};

*/

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// Example of suitable version of _xml_generated_base_</xsl:text><xsl:value-of select="$generatedBaseVersion"/><xsl:text>.h:
//////////////////////////////////////////////////////////////////////////
/*
#pragma once
#include &quot;_xml_generated_base_NP_</xsl:text><xsl:value-of select="$generatedBaseNPVersion"/><xsl:text>.h&quot;

template&lt;typename RealT, typename PLT&gt;
struct _xml_generated_base_</xsl:text><xsl:value-of select="$generatedBaseVersion"/><xsl:text> : public _xml_generated_base_NP_</xsl:text><xsl:value-of select="$generatedBaseNPVersion"/><xsl:text> {
public:	
	typedef RealT real_t;
	typedef PLT PL;

	//////////////////////////////////////////////////////////////////////////
	static constexpr real_t learningRate = PL::learningRate;
	static constexpr real_t dropoutAlive = PL::dropoutAlive;
	static_assert(dropoutAlive &gt; real_t(0.) &amp;&amp; dropoutAlive &lt;=real_t(1.), "Invalid dropoutAlive value! Must be inside of (0,1] interval");
	static constexpr bool bAdjustNcToDropoutRate = PL::bAdjustNcToDropoutRate;
	static constexpr real_t ncMultiplier = PL::ncMultiplier;
	
protected:
	static constexpr neurons_count_t _pos_round(const real_t v)noexcept{
		return neurons_count_t(v) == neurons_count_t(v + real_t(.5)) ? neurons_count_t(v) : neurons_count_t(v + real_t(.5));
	}
	static constexpr bool _hasDropout()noexcept{
		return dropoutAlive &gt; real_t(.0) &amp;&amp; dropoutAlive &lt; real_t(1.0);
	}
	static constexpr neurons_count_t _getNC(const neurons_count_t nc)noexcept{
		return _pos_round( real_t(nc) * ncMultiplier / (
			bAdjustNcToDropoutRate &amp;&amp; _hasDropout() ? dropoutAlive : real_t(1.)
			));
	}
	template&lt;typename NcT&gt; static constexpr neurons_count_t _getNCnoMul(const NcT nc)noexcept{
		return _pos_round( real_t(nc) / (
			bAdjustNcToDropoutRate &amp;&amp; _hasDropout() ? dropoutAlive : real_t(1.)
			));
	}
};

*/
</xsl:text>
</xsl:template>	

</xsl:stylesheet>
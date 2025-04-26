/*********************************************************************/
/*                  						     */
/* Purpose: RSU Distribution Procedure		 		     */
/*	Internation -  A W A R D S 				     */
/*                                                                   */
/*  Org. Date: 02/19/2008        EPAK                                */
/*  Mod. Date: 00/00/0000        XXXX                                */
/*                                                                   */
/*********************************************************************/
Declare			@SharePrice		Float,
			@GrantDate		Char(10)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Set			@SharePrice	= 42.15		---- Wyeth Stock Price
Set			@GrantDate	= '04/27/2006'	---- Wyeth Grant Date (Format: MM/DD/YYYY)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Select	SUBSTRING(OPT.OPT_ID,1,1) as UFC,
	OPT.address6 As GID,
	RTRIM(ISNULL(OPT.NAME_FIRST,'')) As FirstName,
	RTRIM(ISNULL(OPT.NAME_MI,'')) As Middle,
	RTRIM(ISNULL(OPT.NAME_LAST,''))As LastName,
	ISNULL(RTRIM(OPT.E_Mail),'') As Email,
	RTRIM(ADDRESS1) as Address1,
	RTRIM(ISNULL(ADDRESS2,'')) as Address2,
	RTRIM(CITY) as City,
	RTRIM(STATE) as State,
	RTRIM(ZIP) as ZIPCode,
	RTRIM(COUNTRY) As Country,
	OPT.SUB_CD AS DIVOP,
	Convert(char(10),OPT.HIRE_DT,101) As HireDate,
	RTRIM(OPT.USER_CD3) As SpecialHandling,
	PZ.PLAN_ID,
	case
	 when tgr.GRANT_USER_DT is null then ''
	 when tgr.GRANT_USER_DT is not null then convert(char(10),isnull(tgr.GRANT_USER_DT,''),101)
	end  As GrantUserDate,
	OPT.USER_CD1 As GEO,
	geo.ITEM_DESC As GEODesc,
	OPT.USER_CD2 As HYPERION,
	Convert(char(10),tgr.grant_dt,101) AS GRANT_DATE,
	Case TGR.plan_type 
	 when 2 then 'PSA'
	 When 3 then 'RSU'
	 When 4 then 'RSU'
	 When 5 then 'Key'
	End  As PlanType,
	Convert(Char(10),tgr.VEST_DT,101) As Vest_Date,
	RTRIM(isnull(term.TERM_ID,'')) As Term_ID,
	ISNULL(Convert(char(10),OPT.TERM_DT,101),'') AS TERM_DATE,
	ISNULL(tcd.Item_Cd,'') As TaxRate,
	TGR.OPTS_GRNTD As GrossShares,
	@SharePrice As SharePrice,
	TGR.OPTS_GRNTD * @SharePrice As AdjustedIncome

From 	md02n49.wyeth.dbo.optionee opt INNER JOIN md02n49.wyethaux.dbo.tblGrantzRestricted tgr
ON 	tgr.opt_num = opt.opt_num
	INNER JOIN md02n49.wyeth.dbo.user1 geo
ON	opt.USER_CD1 = geo.ITEM_CD
	LEFT JOIN md02n49.wyeth.dbo.term term
ON	term.TERM_ID = tgr.TERM_CD
	LEFT JOIN wyeth.dbo.planz PZ
ON	PZ.PLAN_NUM = tgr.PLAN_NUM
	LEFT JOIN md02n49.WYETH.dbo.TaxAlloc taw
ON     	taw.OPT_NUM   = opt.OPT_NUM
	LEFT JOIN md02n49.WYETH.dbo.TaxCodes tcd
ON     	tcd.CODE_NUM  = taw.TAXCODE_NUM

Where 	TGR.Plan_Type IN ('3','4')
And	Convert(char(10),tgr.grant_dt,101) = @GrantDate
and	(substring(OPT.OPT_ID,1,1) in ('f','c'))

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
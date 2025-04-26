if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_Restricted_WestPort]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_Restricted_WestPort]
GO
CREATE Proc P_Restricted_WestPort
As
/*********************************************************************/
/*                 XXXXXXXX -  Executed by DTS DistributionWestport  */
/*      							     */
/* Purpose: Handling of Restricted Lapse/Distribution 		     */
/*		Data From WestPort				     */
/*   		                                                     */
/*  Org. Date: 03/05/2008	EPAK				     */
/*  Mod. Date: 09/01/2008	EPAK	                             */
/*     Note: Add Total Def Gross & Tax/Fee Shares		     */
/*   		                                                     */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*         		Declare Local Variables			     */ 
/*********************************************************************/
Declare		@Zero			Int,
		@ErrorCode		Int,
		@ARejectCounter		Int,
		@RejectCounter		Int,
		@BRejectCounter		Int,
		@AAcceptCounter		Int,
		@AcceptCounter		Int,
		@BAcceptCounter		Int,
		@NA			Int,
		@PSA			Int,
		@RSU			Int,
		@KEY			Int,
		@DEF			Int,
		@GrantDate		Char(10),
		@CheckTaxCode		Char(8),
		@TLapseShare		Float,
		@TCompIncome		Money,
		@TotalTax		Money,
		@TFee			Money,
		@TTaxShare		Float,
		@TTenderShare		Float,
		@PDefTenderShare	Float,		
		@RDefTenderShare	Float,
		@KDefTenderShare	Float,
		@PDefCounter		Int,
		@RDefCounter		Int,
		@KDefCounter		Int,
		@PCounter		Int,
		@PLapseShare		Float,
		@PCompIncome		Money,
		@PTotalTax		Money,
		@PFee			Money,
		@PTaxShare		Float,
		@PTenderShare		Float,
		@TDefShare		Float,		----	09/01/2008
		@TDefTaxFeeShare	Float,		----	09/01/2008
		@RCounter		Int,
		@RLapseShare		Float,
		@RCompIncome		Money,
		@RTotalTax		Money,
		@RFee			Money,
		@RTaxShare		Float,
		@RTenderShare		Float,
		@KCounter		Int,
		@KLapseShare		Float,
		@KCompIncome		Money,
		@KTotalTax		Money,
		@KFee			Money,
		@KTaxShare		Float,
		@KTenderShare		Float,
		@PlanType		Char(3),
		@OptGrant		Float,
		@OptCan			Float,
		@Process		Varchar(500),
		@ErrorMsg		Varchar(250),
		@ErrorFlag		Int,
		@Rundate		Char(10),
		@FileDate		Char(8),
		@OptionID		Varchar(20),
		@GlobalID		Varchar(20),
		@GrantID		Int,
		@PlanID			Varchar(10),
		@ExerGroup		Varchar(10),
		@ExerID			Varchar(13),
		@DispID			Varchar(13),
		@ExerDate		Char(10),
		@OptExer		Float,
		@MarketPrice		Float,
		@BrokerCode		Varchar(8),
		@SalePrice		Float,
		@TotSaleSold		Float,
		@ExerType		Char(1),
		@SARShare		Float,
		@SARCash		Float,
		@Fee1			Float,
		@Fee2			Float,
		@Fee3			Float,		
		@Commission		Float,
		@SecFee			Float,
		@TaxableComp		Float,
		@TransType		Char(1),
		@TaxCode		Varchar(8),
		@FedRate		Float,
		@StateRate		Float,
		@Local1Rate		Float,
		@Local2Rate		Float,
		@SSRate			Float,
		@MedicareRate		Float,
		@FedDue			Float,
		@StateDue		Float,
		@Local1Due		Float,
		@Local2Due		Float,
		@SSDue			Float,
		@MedicareDue		Float,
		@FedPaid		Float,
		@StatePaid		Float,
		@Local1Paid		Float,
		@Local2Paid		Float,
		@SSPaid			Float,
		@MedicarePaid		Float,
		@ShareTax		Float,		
		@OrderID		Varchar(26),
		@Local3Rate		Float,
		@Local4Rate		Float,
		@Local5Rate		Float,
		@Local6Rate		Float,
		@Local3Due		Float,
		@Local4Due		Float,
		@Local5Due		Float,
		@Local6Due		Float,
		@Local3Paid		Float,
		@Local4Paid		Float,
		@Local5Paid		Float,
		@Local6Paid		Float,
		@ShareTender		Float,
		@OverrideTax		Float,
		@ActualTax		Float,
		@Name			Varchar(20),
		@UserCode		Varchar(5),
		@Not			Char(3)

/*********************************************************************/
/*    	      Assign Local Variables with Initial Value		     */ 
/*********************************************************************/
Select	@Zero			= 0,
	@Process		= 'P_Restricted_WestPort',
	@Not			= 'N/A',
	@RunDate 		= 
	REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Month,Getdate()))))) + RTRIM(Convert(Char(2),DATEPART(Month,Getdate()))) + '-' +
	REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Day,Getdate()))))) + RTRIM(Convert(Char(2),DATEPART(Day,Getdate()))) + '-' +
	REPLICATE('0', 4 - DATALENGTH(RTRIM(Convert(Char(4),DATEPART(Year,Getdate()))))) + RTRIM(Convert(Char(4),DATEPART(Year,Getdate()))),
	@TLapseShare		= 0,
	@TCompIncome		= 0,
	@TotalTax		= 0,
	@TFee			= 0,
	@TTaxShare		= 0,
	@TTenderShare		= 0,
	@PDefTenderShare	= 0,
	@RDefTenderShare	= 0,
	@KDefTenderShare	= 0,
	@PDefCounter		= 0,
	@RDefCounter		= 0,
	@KDefCounter		= 0,
	@PLapseShare		= 0,
	@PCompIncome		= 0,
	@PTotalTax		= 0,
	@PFee			= 0,
	@PTaxShare		= 0,
	@PTenderShare		= 0,
	@RLapseShare		= 0,
	@RCompIncome		= 0,
	@RTotalTax		= 0,
	@RFee			= 0,
	@RTaxShare		= 0,
	@RTenderShare		= 0,
	@KLapseShare		= 0,
	@KCompIncome		= 0,
	@KTotalTax		= 0,
	@KFee			= 0,
	@KTaxShare		= 0,
	@KTenderShare		= 0,
	@NA			= 0,
	@PSA			= 0,
	@RSU			= 0,
	@KEY			= 0,
	@DEF			= 0,
	@AcceptCounter		= 0,
	@RejectCounter		= 0,
	@PCounter		= 0,
	@RCounter		= 0,
	@KCounter		= 0,
	@ErrorFlag		= 0,
	@TDefShare		= 0,
	@TDefTaxFeeShare	= 0

/*********************************************************************/
/*              					             */
/*********************************************************************/
Insert	ProcessLog
Values(Getdate(),@Process,@Zero,'SUCCESS','START - WestPort Lapse/Distribution Data From WestPort')

/*********************************************************************/
/*         	Open Transaction				     */ 
/*********************************************************************/
BEGIN TRAN WestPortRestricted

/*********************************************************************/
/*         							     */ 
/*********************************************************************/
Truncate Table tblDistributionRejectReport
	
	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg 	= '1001 - tblDistributionRejectReport Table Delete Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
/*         							     */ 
/*********************************************************************/
Truncate Table tblDistributionAcceptReport

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '2001 - tblDistributionAcceptReport Table Delete Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
/*         	Accepted Report Reporting			     */
/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	REPLICATE(' ',55) + 'Restricted Lapse/Distribution Data From Westport Accepted Report'

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3001 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3002 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error	
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	'Run Date: ' + @Rundate

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3003 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error	
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3004 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3005 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	  REPLICATE(' ',1)  + 'Global ID'
	+ REPLICATE(' ',5)  + 'Option ID'
	+ REPLICATE(' ',3)  + 'First/Last Name'
	+ REPLICATE(' ',6)  + 'Grant Date'
	+ REPLICATE(' ',2)  + 'Type'
	+ REPLICATE(' ',1)  + 'Exercise ID' 
	+ REPLICATE(' ',4)  + 'Gross Shares' 
	+ REPLICATE(' ',3)  + 'Comp. Income' 
	+ REPLICATE(' ',8)  + 'Total Taxes' 
	+ REPLICATE(' ',5)  + 'Total Fees' 
	+ REPLICATE(' ',3)  + 'Tax/Fee Shares' 
	+ REPLICATE(' ',2)  + 'Net Shares'

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3006 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionAcceptReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '3007 - tblDistributionAcceptReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Select	@BAcceptCounter = Count(*) From tblDistributionAcceptReport

/*********************************************************************/
/*         	Rejected Report Reporting			     */
/*********************************************************************/
Insert	tblDistributionRejectReport
Select	REPLICATE(' ',35) + 'Restricted Lapse/Distribution Data From Westport Rejected Report'

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4001 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4002 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	'Control Date: ' + @Rundate

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4003 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4004 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4005 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	  REPLICATE(' ',1)  + 'Global ID'
	+ REPLICATE(' ',4)  + 'Option ID'
	+ REPLICATE(' ',4)  + 'First/Last Name'
	+ REPLICATE(' ',7)  + 'Grant Date'
	+ REPLICATE(' ',2)  + 'Type'
	+ REPLICATE(' ',1)  + 'Exercise ID' 
	+ REPLICATE(' ',3)  + 'Grant NUM' 
	+ REPLICATE(' ',7)  + 'Error Description' 

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4006 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Insert	tblDistributionRejectReport
Select	REPLICATE(' ',1)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '4007 - tblDistributionRejectReport Table INSERT Failure!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
Select	@BRejectCounter = Count(*) From tblDistributionRejectReport

/*********************************************************************/
/*         INTO TEMP Table					     */
/*********************************************************************/
Select	CASE
	 WHEN LEN(RTRIM(RTRIM(OPT.address6))) > @Zero 
	 	THEN RTRIM(RTRIM(OPT.address6))
	 ELSE NULL
	END								As Global_ID,
	CONVERT(Int,RTRIM(SUBSTRING(DRI.Import,22,13)))			As Grant_Number,
	RTRIM(SUBSTRING(DRI.Import,36,10)) 				As Plan_ID,
	RTRIM(SUBSTRING(DRI.Import,47,10)) 				As Exercise_Group,
	RTRIM(SUBSTRING(DRI.Import,58,13)) 				As Exercise_ID,
	RTRIM(SUBSTRING(DRI.Import,72,13)) 				As Dispositiopn_ID,
	RTRIM(SUBSTRING(DRI.Import,86,4)) + '-' +
	RTRIM(SUBSTRING(DRI.Import,90,2)) + '-' +
	RTRIM(SUBSTRING(DRI.Import,92,2))				As Exercise_Date,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,95,15))))/1000000 	As Options_Exercised,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,111,15))))/1000000 	As Market_Price,
	RTRIM(SUBSTRING(DRI.Import,127,8)) 				As Broker_Code,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,136,15))))/1000000 	As Sale_Price,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,152,15))))/1000000 	As Total_Shares_Sold,
	RTRIM(SUBSTRING(DRI.Import,168,1)) 				As Exercise_Type,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,170,15))))/1000000 	As SAR_Shares,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,186,15))))/1000000 	As SAR_Cash,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,202,15))))/1000000 	As Fee_1,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,218,15))))/1000000 	As Fee_2,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,234,15))))/1000000 	As Fee_3,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,250,15))))/1000000 	As Commission,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,266,15))))/1000000 	As Sec_Fee,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,282,15))))/1000000 	As Taxable_Compensation,
	RTRIM(SUBSTRING(DRI.Import,298,1)) 				As Transaction_Type,
	RTRIM(SUBSTRING(DRI.Import,300,8)) 				As Tax_Code,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,309,15))))/1000000 	As Fed_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,325,15))))/1000000 	As State_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,341,15))))/1000000 	As Local1_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,357,15))))/1000000 	As Local2_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,373,15))))/1000000 	As SS_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,389,15))))/1000000 	As Medicare_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,405,15))))/1000000 	As Fed_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,421,15))))/1000000 	As State_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,437,15))))/1000000 	As Local1_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,453,15))))/1000000 	As Local2_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,469,15))))/1000000 	As SS_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,485,15))))/1000000 	As Medicare_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,501,15))))/1000000 	As Fed_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,517,15))))/1000000 	As State_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,533,15))))/1000000 	As Local1_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,549,15))))/1000000 	As Local2_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,565,15))))/1000000 	As SS_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,581,15))))/1000000 	As Medicare_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,597,15))))/1000000 	As Shares_Of_Taxes,
	RTRIM(SUBSTRING(DRI.Import,613,26))				As Order_ID,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,640,15))))/1000000 	As Local3_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,656,15))))/1000000 	As Local4_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,672,15))))/1000000 	As Local5_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,688,15))))/1000000 	As Local6_Rate,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,704,15))))/1000000 	As Local3_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,720,15))))/1000000 	As Local4_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,736,15))))/1000000 	As Local5_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,752,15))))/1000000 	As Local6_Due,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,768,15))))/1000000 	As Local3_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,784,15))))/1000000 	As Local4_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,800,15))))/1000000 	As Local5_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,816,15))))/1000000 	As Local6_Paid,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,832,15))))/1000000 	As Shares_Tendered,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,848,15))))/1000000 	As Override_Tax_Price,
	(CONVERT(Float,RTRIM(SUBSTRING(DRI.Import,864,15))))/1000000 	As Actual_Tax_Price,
	RTRIM(SUBSTRING(DRI.Import,1,20))				As Optionee_ID,
	GrantType =
	CASE
	 WHEN TGR.PLAN_TYPE = '2' THEN RTRIM('PSA')
	 WHEN TGR.PLAN_TYPE = '3' THEN RTRIM('RSU')
	 WHEN TGR.PLAN_TYPE = '4' THEN RTRIM('RSU')
	 WHEN TGR.PLAN_TYPE = '5' THEN RTRIM('KEY')
	 WHEN TGR.PLAN_TYPE IS NULL THEN NULL
	 ELSE @Not
	END,
	ISNULL(TGR.OPTS_GRNTD,0)					As Restricted_Option_Grant,
	ISNULL(TGR.CANCEL1_SHARES,0)					As Restrict_Cancel_Shares,
	TC.ITEM_CD							As CheckTaxCode,
	REPLICATE('0', 4 - DATALENGTH(RTRIM(Convert(Char(4),DATEPART(Year,TGR.GRANT_DT))))) + RTRIM(Convert(Char(4),DATEPART(Year,TGR.GRANT_DT))) + '-' +
	REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Month,TGR.GRANT_DT))))) + RTRIM(Convert(Char(2),DATEPART(Month,TGR.GRANT_DT))) + '-' +
	REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Day,TGR.GRANT_DT))))) + RTRIM(Convert(Char(2),DATEPART(Day,TGR.GRANT_DT)))
	As GrantDate,
	LEFT(LEFT(OPT.NAME_FIRST,1) + SPACE(1) + RTRIM(OPT.NAME_LAST),20) As FirstLastName,
	UserCode =
	CASE
	 WHEN OPT.USER_CD3 Like '%FSE%' THEN OPT.USER_CD3
	 ELSE NULL
	END

INTO	#ErikPak
From	WyethAux.dbo.tblDistributionRestrictedImport DRI LEFT JOIN Wyeth.dbo.Optionee OPT
ON	RTRIM(SUBSTRING(DRI.Import,1,20)) = RTRIM(OPT.OPT_ID)
	LEFT JOIN tblGrantzRestricted TGR
ON	CONVERT(Int,RTRIM(SUBSTRING(DRI.Import,22,13)))	= TGR.GRANT_NUM
	LEFT JOIN Wyeth.dbo.TaxCodes TC
ON	RTRIM(SUBSTRING(DRI.Import,300,8)) = RTRIM(TC.ITEM_CD)

	Select @ErrorCode = @@ERROR	
	IF @ErrorCode <> @Zero
	 Begin
		SET @ErrorMsg = '5000 - Internal TEMP Table Creation Error!!!'
		SET @ErrorFlag	= 999
		GOTO Header_Footeer_Error
	 End

/*********************************************************************/
/*         							     */
/*********************************************************************/
DECLARE DCheck_CURSOR CURSOR FOR
Select	*
From	#ErikPak
ORDER BY GrantType

/*********************************************************************/
/*                   OPEN Cursor                                     */ 
/*********************************************************************/
OPEN DCheck_CURSOR

FETCH NEXT FROM DCheck_CURSOR INTO 	@GlobalID,
					@GrantID,
					@PlanID,
					@ExerGroup,
					@ExerID,
					@DispID,
					@ExerDate,
					@OptExer,
					@MarketPrice,
					@BrokerCode,
					@SalePrice,
					@TotSaleSold,
					@ExerType,
					@SARShare,
					@SARCash,
					@Fee1,
					@Fee2,
					@Fee3,
					@Commission,
					@SecFee,
					@TaxableComp,
					@TransType,
					@TaxCode,
					@FedRate,
					@StateRate,
					@Local1Rate,
					@Local2Rate,
					@SSRate,
					@MedicareRate,
					@FedDue,
					@StateDue,
					@Local1Due,
					@Local2Due,
					@SSDue,
					@MedicareDue,
					@FedPaid,
					@StatePaid,
					@Local1Paid,
					@Local2Paid,
					@SSPaid,
					@MedicarePaid,
					@ShareTax,
					@OrderID,
					@Local3Rate,
					@Local4Rate,
					@Local5Rate,
					@Local6Rate,
					@Local3Due,
					@Local4Due,
					@Local5Due,
					@Local6Due,
					@Local3Paid,
					@Local4Paid,
					@Local5Paid,
					@Local6Paid,
					@ShareTender,
					@OverrideTax,
					@ActualTax,
					@OptionID,
					@PlanType,
					@OptGrant,
					@OptCan,
					@CheckTaxCode,
					@GrantDate,
					@Name,
					@Usercode

/*********************************************************************/
/*              					             */
/*********************************************************************/
WHILE (@@FETCH_STATUS <> -1)

  BEGIN
	IF (@@FETCH_STATUS <> -2)
	 Begin
	/*************************************************************/
	/*             					             */
	/*************************************************************/
	IF LTRIM(RTRIM(@ExerDate)) = '--'
	 Begin
		SET @ExerDate = 'N/A'
	 End

	/*************************************************************/
	/*  Include ONLY PlanType IN (2,3,4,5) From Grant Restricted */
	/*  @OptionID From Optionee Table			     */
	/*  Exercise ID & Grant Num MUST Be Unique		     */
	/*************************************************************/
	IF (@PlanType IS NULL) OR (@GlobalID IS NULL) OR 
		(Select Count(EXER_ID) From tblDistributionRestricted Where RTRIM(EXER_ID) = RTRIM(@ExerID)) <> @Zero
		OR (Select Count(GRANT_NUM) From tblDistributionRestricted Where GRANT_NUM = @GrantID) <> @Zero

	 Begin
		/*****************************************************/
		/*    					             */
		/*****************************************************/
		IF @PlanType IS NULL
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Grant Number NOT Found in tblGrantzRestricted Table'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		 Else IF @Name IS NULL
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'OPT ID NOT Found in EOWIN'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		Else IF @GlobalID IS NULL
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Global ID "UNKNOWN" in EOWIN'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End
		Else IF (Select Count(GRANT_NUM) From tblDistributionRestricted Where GRANT_NUM = @GrantID) <> @Zero
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Duplicate "Grant Num" Found in tblDistributionRestricted Table'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End
		Else 
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Duplicate "Exercise_ID" Found in tblDistributionRestricted Table'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
	 End
	ELSE
	 Begin
		/*****************************************************/
		/*    					             */
		/*****************************************************/
		IF @PlanType = @Not
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Invalid Plan Type in tblGrantzRestricted Table'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*   Due to Multiplier PSA Option Grant Differ       */
		/*****************************************************/
		IF @OptGrant <> @OptExer And @PlanType <> 'PSA'
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Shares Granted DOES NOT EQUAL "OPTS_GRNTD" in tblGrantzRestricted Table'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End 

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF ((@OptGrant <= @OptCan) And (@PlanType = 'PSA'))
			     OR ((@OptExer > (@OptGrant - @OptCan)) And (@PlanType <> 'PSA'))
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Share(s) Canceled'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF ISDATE(@ExerDate) = @Zero
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Invalid Distribution Date - [' + RTRIM(@ExerDate) + ']'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End 
		
		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF CONVERT(DateTime,@ExerDate) > CONVERT(DateTime,@RunDate)
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Future Distribution Date - [' + RTRIM(@ExerDate) + ']'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF @CheckTaxCode IS NULL And @Taxcode <> ''
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Invalid Tax Code'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF ((@Fee2 + @Fee3 + @Commission + @SecFee) <> @Zero)
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Fee-2 OR Fee-3 OR Commission OR Sec-Fee DOES NOT Equals Zero'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End
		
		/*****************************************************/
		/* 	ONLY RSU Can Have Taxable Comp = Zero        */
		/*        	100% Deferral			     */
		/*****************************************************/
		ELSE IF @TaxableComp = @Zero And (@PlanType <> 'RSU' Or @ShareTender <> @Zero)
		 Begin

			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Taxable Compensation Equals Zero'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

--4/25/2008		/*****************************************************/
--4/25/2008		/* ONLY RSU - Tax Comp And/Or ShareTax And/Or Fee 1  */
--4/25/2008		/*            <> Zero 				     */
--4/25/2008		/*****************************************************/
--4/25/2008		ELSE IF @ShareTender = @Zero And @PlanType = 'RSU' And (@TaxableComp <> @Zero Or @ShareTax <> @Zero Or @Fee1 <> @Zero)
--4/25/2008		 Begin
--4/25/2008			/*********************************************/
--4/25/2008			/*   				             */
--4/25/2008			/*********************************************/
--4/25/2008			SET @ErrorMsg = 'Taxable Compensation And/Or Tax Shares And/Or Fee 1 NOT Equals Zero for RSU Deferral'
--4/25/2008			SET @ErrorFlag	= 999
--4/25/2008			GOTO Data_Error_Check
--4/25/2008
--4/25/2008			/*********************************************/
--4/25/2008			/*   				             */
--4/25/2008			/*********************************************/
--4/25/2008		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF CONVERT(Decimal(15,2),(@FedRate + @StateRate + @SSRate + @MedicareRate + @Local1Rate + @Local2Rate + @Local3Rate + @Local4Rate + @Local5Rate + @Local6Rate)) >= 100.00
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Total Tax Rate Equals or Greater than 100'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End


		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF (@ExerType) <> '3'     ---- Restricted Lapse
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Transaction Type DOES NOT EQUAL "3" (Restricted Lapse)'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF CONVERT(Decimal(15,2),(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid)) > CONVERT(Decimal(15,2),@TaxableComp)
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Tax Paid is GREATER THAN Taxable Compensation'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF (CONVERT(Decimal(15,2),(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid)) = CONVERT(Decimal(15,2),@TaxableComp))
			And @ShareTender <> @Zero
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Tax Paid EQUALS Taxable Compensation But Share Tendered NOT Zero'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF @ShareTax >= @OptExer
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Gross Shares Less Than or Equal To Fee/Tax Shares'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF @ShareTender > @OptExer
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Gross Shares LESS THAN Net Shares'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF ((@ShareTender <> @Zero) And (@ShareTender <> (@OptExer - @ShareTax)))
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'Gross Shares minus Tax/Fee Shares NOT EQUAL Net Shares'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE IF @ExerID = ''
		 Begin
			/*********************************************/
			/*   				             */
			/*********************************************/
			SET @ErrorMsg = 'BLANK Exercise ID'
			SET @ErrorFlag	= 999
			GOTO Data_Error_Check

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
		ELSE
		 Begin
			/*********************************************/
			/*   	Accept Report Generation             */
			/*********************************************/
			Insert	tblDistributionAcceptReport
			Select	RTRIM(@GlobalID) + REPLICATE(' ',ABS(LEN(RTRIM(@GlobalID)) - 14)) +
				RTRIM(@OptionID) + REPLICATE(' ',ABS(LEN(RTRIM(@OptionID)) - 12)) +
				RTRIM(@Name) + REPLICATE(' ',ABS(LEN(RTRIM(@Name)) - 22)) +
				RTRIM(@GrantDate) + REPLICATE(' ',ABS(LEN(RTRIM(@GrantDate)) - 12)) +
				RTRIM(@PlanType) + REPLICATE(' ',ABS(LEN(RTRIM(@PlanType)) - 5)) +
				RTRIM(@ExerID) + REPLICATE(' ',ABS(LEN(RTRIM(@ExerID)) - 15)) +
				RTRIM(WyethAux.dbo.fnAddCommaToNumber(@OptExer)) + REPLICATE(' ',ABS(LEN(RTRIM(WyethAux.dbo.fnAddCommaToNumber(@OptExer))) - 15)) +
				LTRIM(RTRIM( CONVERT(Char,(CONVERT(Money,@TaxableComp)),1))) + REPLICATE(' ',ABS(LEN(LTRIM(RTRIM(CONVERT(Char,(CONVERT(Money,@TaxableComp)),1)))) - 20)) +
				LTRIM(RTRIM(CONVERT(Char,(CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid))),1)))
				+ REPLICATE(' ',ABS(LEN(LTRIM(RTRIM(CONVERT(Char,CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid)),1)))) - 16)) +
				LTRIM(RTRIM(CONVERT(Char,CONVERT(Money,@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee),1)))
				+ REPLICATE(' ',ABS(LEN(LTRIM(RTRIM(CONVERT(Char,CONVERT(Money,@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee),1)))) - 16)) +
				RTRIM(WyethAux.dbo.fnAddCommaToNumber(@ShareTax)) + REPLICATE(' ',ABS(LEN(RTRIM(WyethAux.dbo.fnAddCommaToNumber(@ShareTax))) - 14)) +
				RTRIM(WyethAux.dbo.fnAddCommaToNumber(@ShareTender)) + REPLICATE(' ',ABS(LEN(RTRIM(WyethAux.dbo.fnAddCommaToNumber(@ShareTender))) - 13)) +
				CASE
				 WHEN @ShareTender = @Zero THEN RTRIM('*DEF*') + REPLICATE(' ',ABS(LEN(RTRIM('*DEF*')) - 7))
				 ELSE REPLICATE(' ',ABS(LEN(RTRIM(' ')) - 7))
				END +
				CASE
				 WHEN @Usercode IS NOT NULL THEN RTRIM('*' + RTRIM(@Usercode) + '*') + REPLICATE(' ',ABS(LEN(RTRIM('*' + RTRIM(@Usercode) + '*')) - 6))
				 ELSE REPLICATE(' ',ABS(LEN(RTRIM(' ')) - 6))
				END

				Select @ErrorCode = @@ERROR	
				IF @ErrorCode <> @Zero
				 Begin
					ROLLBACK TRAN WestPortRestricted
					Insert	ProcessLog
					Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblDistributionAcceptReport Table INSERT Failure!!!')
					RAISERROR (70001,16,2)
					DROP TABLE #ErikPak
					CLOSE DCheck_CURSOR
					DEALLOCATE DCheck_CURSOR
					RETURN @ErrorCode		
				 End

			/*********************************************/
			/*   	   Total Tally		             */
			/*********************************************/
			Select	@TLapseShare	= @TLapseShare 	+ @OptExer,
				@TCompIncome	= @TCompIncome 	+ CONVERT(Money,(@TaxableComp),2),
				@TotalTax	= @TotalTax	+ CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid),2),
				@TFee		= @TFee		+ CONVERT(Money,(@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee)),
				@TTaxShare	= @TTaxShare	+ @ShareTax,
				@TTenderShare	= @TTenderShare	+ @ShareTender

			/*********************************************/
			/*           DEF Counter & Tally             */
			/*********************************************/
			IF @ShareTender = @Zero
			 Begin
				IF @PlanType = 'RSU'
				 Begin
					Select	@RDefTenderShare = @RDefTenderShare + (@OptExer - @ShareTax),
						@RDefCounter = @RDefCounter + 1
				 End

				IF @PlanType = 'PSA'
				 Begin
					Select	@PDefTenderShare = @PDefTenderShare + (@OptExer - @ShareTax),
						@PDefCounter = @PDefCounter + 1
				 End

				IF @PlanType = 'KEY'
				 Begin
					Select	@KDefTenderShare = @KDefTenderShare + (@OptExer - @ShareTax),
						@KDefCounter = @KDefCounter + 1
				 End


				/*************************************/
				/*    DEF Gross Tally  - 09/01/2008  */
				/*************************************/
				Select	@TDefShare		=	@TDefShare + @OptExer,
					@TDefTaxFeeShare	=	@TDefTaxFeeShare + @ShareTax
				
				/*************************************/
				/*    DEF Gross Tally  - 09/01/2008  */
				/*************************************/
				Select @DEF = @DEF + 1
			 End

			/*********************************************/
			/*   		RSU Tally	             */
			/*********************************************/
			IF @PlanType = 'RSU'
			 Begin
				/**************************************/
				Select	@RLapseShare	= @RLapseShare 	+ @OptExer,
					@RCompIncome	= @RCompIncome 	+ CONVERT(Money,(@TaxableComp),2),
					@RTotalTax	= @RTotalTax	+ CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid),2),
					@RFee		= @RFee		+ CONVERT(Money,(@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee)),
					@RTaxShare	= @RTaxShare	+ @ShareTax,
					@RTenderShare	= @RTenderShare	+ @ShareTender,
					@RCounter	= @RCounter	+ 1
				/**************************************/
			 End

			/*********************************************/
			/*   		PSA Tally	             */
			/*********************************************/
			IF @PlanType = 'PSA'
			 Begin
				/**************************************/
				Select	@PLapseShare	= @PLapseShare 	+ @OptExer,
					@PCompIncome	= @PCompIncome 	+ CONVERT(Money,(@TaxableComp),2),
					@PTotalTax	= @PTotalTax	+ CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid),2),
					@PFee		= @PFee		+ CONVERT(Money,(@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee)),
					@PTaxShare	= @PTaxShare	+ @ShareTax,
					@PTenderShare	= @PTenderShare	+ @ShareTender,
					@PCounter	= @PCounter	+ 1
				/**************************************/
			 End

			/*********************************************/
			/*   		KEY Tally	             */
			/*********************************************/
			IF @PlanType = 'KEY'
			 Begin
				/**************************************/
				Select	@KLapseShare	= @KLapseShare 	+ @OptExer,
					@KCompIncome	= @KCompIncome 	+ CONVERT(Money,(@TaxableComp),2),
					@KTotalTax	= @KTotalTax	+ CONVERT(Money,(@FedPaid + @StatePaid + @SSPaid + @MedicarePaid + @Local1Paid + @Local2Paid + @Local3Paid + @Local4Paid + @Local5Paid + @Local6Paid),2),
					@KFee		= @KFee		+ CONVERT(Money,(@Fee1 + @Fee2 + @Fee3 + @Commission + @SecFee)),
					@KTaxShare	= @KTaxShare	+ @ShareTax,
					@KTenderShare	= @KTenderShare	+ @ShareTender,
					@KCounter	= @KCounter	+ 1
				/**************************************/
			 End

			/*********************************************/
			/*   		INSERT Validated Data        */
			/*********************************************/
			Insert	tblDistributionRestricted
				(
					GRANT_NUM,OPT_ID,EXER_ID,EXER_DT,TRANS_TYPE,OPTS_EXER,MKT_PRICE,TXBL_COMP,
					TAX_CODE,TAX_SHRS,SHRS_TEND,FIXED_FEE1,FIXED_FEE2,FIXED_FEE3,COMMISSION,SEC_FEE,
					FED_RATE,STATE_RATE,LOC1_RATE,LOC2_RATE,LOC3_RATE,LOC4_RATE,LOC5_RATE,
					LOC6_RATE,SOSEC_RATE,MED_RATE,FED_DUE,STATE_DUE,LOC1_DUE,LOC2_DUE,
					LOC3_DUE,LOC4_DUE,LOC5_DUE,LOC6_DUE,SOSEC_DUE,MED_DUE,FED_PAID,STATE_PAID,
					LOC1_PAID,LOC2_PAID,LOC3_PAID,LOC4_PAID,LOC5_PAID,LOC6_PAID,SOSEC_PAID,
					MED_PAID
				)
			Select	@GrantID,@OptionID,@ExerID,@ExerDate,@TransType,@OptExer,@MarketPrice,@TaxableComp,
				@TaxCode,@ShareTax,@ShareTender,@Fee1,@Fee2,@Fee3,@Commission,@SecFee,
				@FedRate,@StateRate,@Local1Rate,@Local2Rate,@Local3Rate,@Local4Rate,@Local5Rate,
				@Local6Rate,@SSRate,@MedicareRate,@FedDue,@StateDue,@Local1Due,@Local2Due,
				@Local3Due,@Local4Due,@Local5Due,@Local6Due,@SSDue,@MedicareDue,@FedPaid,@StatePaid,
				@Local1Paid,@Local2Paid,@Local3Paid,@Local4Paid,@Local5Paid,@Local6Paid,@SSPaid,
				@MedicarePaid

				Select @ErrorCode = @@ERROR	
				IF @ErrorCode <> @Zero
				 Begin
					ROLLBACK TRAN WestPortRestricted
					Insert	ProcessLog
					Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblDistributionRestricted Table INSERT Failure!!!')
					RAISERROR (70001,16,3)
					DROP TABLE #ErikPak
					CLOSE DCheck_CURSOR
					DEALLOCATE DCheck_CURSOR
					RETURN @ErrorCode		
				 End

			/*********************************************/
			/*   				             */
			/*********************************************/
			IF @TaxableComp = @Zero
			 Begin
				Update	tblDistributionRestricted
				SET	SAP = 'X'
				Where	GRANT_NUM = @GrantID
				And	OPT_ID = @OptionID
				And	EXER_ID = @ExerID

					Select @ErrorCode = @@ERROR	
					IF @ErrorCode <> @Zero
					 Begin
						ROLLBACK TRAN WestPortRestricted
						Insert	ProcessLog
						Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblDistributionRestricted Table Update Failure!!!')
						RAISERROR (70001,16,3)
						DROP TABLE #ErikPak
						CLOSE DCheck_CURSOR
						DEALLOCATE DCheck_CURSOR
						RETURN @ErrorCode		
					 End
			 End

			/*********************************************/
			/*   				             */
			/*********************************************/
			IF @ShareTender = @Zero
			 Begin 
				Update	tblGrantzRestricted
				SET	[DEFERRED] = 'Y'
				Where	GRANT_NUM = @GrantID

					Select @ErrorCode = @@ERROR	
					IF @ErrorCode <> @Zero
					 Begin
						ROLLBACK TRAN WestPortRestricted
						Insert	ProcessLog
						Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblGrantzRestricted Table UPDATE Failure!!!')
						RAISERROR (70001,16,3)
						DROP TABLE #ErikPak
						CLOSE DCheck_CURSOR
						DEALLOCATE DCheck_CURSOR
						RETURN @ErrorCode		
					 End
			 End

			/*********************************************/
			/*   				             */
			/*********************************************/
		 End

		/*****************************************************/
		/*    					             */
		/*****************************************************/
	 End	

	/*************************************************************/
	/*           ERROR Report Generator		             */
	/*************************************************************/
	Data_Error_Check:
	IF @ErrorFlag <> @Zero
	 Begin

		/*****************************************************/
		/*		Reject Report		             */
		/*****************************************************/
		Insert	tblDistributionRejectReport
		Select	RTRIM(ISNULL(@GlobalID,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@GlobalID,@Not))) - 14)) +
			RTRIM(ISNULL(@OptionID,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@OptionID,@Not))) - 13)) +
			RTRIM(ISNULL(@Name,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@Name,@Not))) - 22)) +
			RTRIM(ISNULL(@GrantDate,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@GrantDate,@Not))) - 12)) +
			RTRIM(ISNULL(@PlanType,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@PlanType,@Not))) - 5)) +
			RTRIM(ISNULL(@ExerID,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@ExerID,@Not))) - 15)) +
			RTRIM(ISNULL(@GrantID,@Not)) + REPLICATE(' ',ABS(LEN(RTRIM(ISNULL(@GrantID,@Not))) - 15)) +
			@ErrorMsg

				Select @ErrorCode = @@ERROR
				IF @ErrorCode <> @Zero
				 Begin
					ROLLBACK TRAN WestPortRestricted
					Insert	ProcessLog
					Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblDistributionRejectReport Table INSERT Failure!!!')
					RAISERROR (70001,16,4)
					DROP TABLE #ErikPak
					CLOSE DCheck_CURSOR
					DEALLOCATE DCheck_CURSOR
					RETURN @ErrorCode		
				 End

		/*****************************************************/
		/*	Reject Record(s) Insert		             */
		/*****************************************************/
		Insert	tblDistributionRestrictedReject
			(
				OptID,GrantNumber,PlanID,ExerciseGroup,ExerciseID,DispositionID,ExerciseDate,OptionExercised,
				MarketPrice,BrokerCode,SalePrice,TotalSharesSold,ExerciseType,SARShares,SARCash,
				Fee1,Fee2,Fee3,Commission,SecFee,TaxableCompensation,TransactionType,TaxCode,FederalRate,
				StateRate,Local1Rate,Local2Rate,SocialSecurityRate,MedicareRate,FederalDue,StateDue,Local1Due,
				Local2Due,SocialSecurityDue,MedicareDue,FederalPaid,StatePaid,Local1Paid,Local2Paid,SocialSecurityPaid,
				MedicarePaid,SharesForTaxes,OrderID,Local3Rate,Local4Rate,Local5Rate,Local6Rate,
				Local3Due,Local4Due,Local5Due,Local6Due,Local3Paid,Local4Paid,Local5Paid,Local6Paid,
				SharesTendered,OverrideTaxPrice,ActualTaxPrice,ErrorMessage
			)
		Select	@OptionID,@GrantID,@PlanID,@ExerGroup,@ExerID,@DispID,@ExerDate,@OptExer,
			@MarketPrice,@BrokerCode,@SalePrice,@TotSaleSold,@ExerType,@SARShare,@SARCash,
			@Fee1,@Fee2,@Fee3,@Commission,@SecFee,@TaxableComp,@TransType,@TaxCode,@FedRate,
			@StateRate,@Local1Rate,@Local2Rate,@SSRate,@MedicareRate,@FedDue,@StateDue,@Local1Due,
			@Local2Due,@SSDue,@MedicareDue,@FedPaid,@StatePaid,@Local1Paid,@Local2Paid,@SSPaid,
			@MedicarePaid,@ShareTax,@OrderID,@Local3Rate,@Local4Rate,@Local5Rate,@Local6Rate,
			@Local3Due,@Local4Due,@Local5Due,@Local6Due,@Local3Paid,@Local4Paid,@Local5Paid,@Local6Paid,
			@ShareTender,@OverrideTax,@ActualTax,@ErrorMsg

				Select @ErrorCode = @@ERROR	
				IF @ErrorCode <> @Zero
				 Begin
					ROLLBACK TRAN WestPortRestricted
					Insert	ProcessLog
					Values(Getdate(),@Process,@ErrorCode,'FAILURE','(Option ID: [' + RTRIM(@OptionID) +  '] And Exercise ID: [' + RTRIM(@ExerID) + ']) - tblDistributionRestrictedReject Table INSERT Failure!!!')
					RAISERROR (70001,16,5)
					DROP TABLE #ErikPak
					CLOSE DCheck_CURSOR
					DEALLOCATE DCheck_CURSOR
					RETURN @ErrorCode		
				 End

			/*********************************************/
			/*		Grant Type Counter           */
			/*********************************************/
			IF @PlanType = 'RSU'
				Select @RSU = @RSU + 1

			IF @PlanType = 'PSA'
				Select @PSA = @PSA + 1

			IF @PlanType = 'KEY'
				Select @KEY = @KEY + 1

			IF @PlanType IS NULL OR @PlanType = @Not
				Select @NA = @NA + 1

			/*********************************************/
			/*				             */
			/*********************************************/
	 End

	/*************************************************************/
	/*            RESET Error Flag 			             */
	/*************************************************************/
	SET @ErrorFlag	= 0

	/*************************************************************/
	/*             					             */
	/*************************************************************/
	 End

	/*************************************************************/
	/*             					             */
	/*************************************************************/
	FETCH NEXT FROM DCheck_CURSOR INTO 	@GlobalID,
						@GrantID,
						@PlanID,
						@ExerGroup,
						@ExerID,
						@DispID,
						@ExerDate,
						@OptExer,
						@MarketPrice,
						@BrokerCode,
						@SalePrice,
						@TotSaleSold,
						@ExerType,
						@SARShare,
						@SARCash,
						@Fee1,
						@Fee2,
						@Fee3,
						@Commission,
						@SecFee,
						@TaxableComp,
						@TransType,
						@TaxCode,
						@FedRate,
						@StateRate,
						@Local1Rate,
						@Local2Rate,
						@SSRate,
						@MedicareRate,
						@FedDue,
						@StateDue,
						@Local1Due,
						@Local2Due,
						@SSDue,
						@MedicareDue,
						@FedPaid,
						@StatePaid,
						@Local1Paid,
						@Local2Paid,
						@SSPaid,
						@MedicarePaid,
						@ShareTax,
						@OrderID,
						@Local3Rate,
						@Local4Rate,
						@Local5Rate,
						@Local6Rate,
						@Local3Due,
						@Local4Due,
						@Local5Due,
						@Local6Due,
						@Local3Paid,
						@Local4Paid,
						@Local5Paid,
						@Local6Paid,
						@ShareTender,
						@OverrideTax,
						@ActualTax,
						@OptionID,
						@PlanType,
						@OptGrant,
						@OptCan,
						@CheckTaxCode,
						@GrantDate,
						@Name,
						@Usercode

  END

/*********************************************************************/
/*              					             */
/*********************************************************************/
CLOSE DCheck_CURSOR
DEALLOCATE DCheck_CURSOR

/*********************************************************************/
/*              House Cleaning				             */
/*********************************************************************/
DROP TABLE #ErikPak

/*********************************************************************/
/*              Total Record Counter			             */
/*********************************************************************/
Select	@AAcceptCounter	= COUNT(*) From tblDistributionAcceptReport
Select	@ARejectCounter	= COUNT(*) From tblDistributionRejectReport

Select	@AcceptCounter = ABS(@AAcceptCounter - @BAcceptCounter),
	@RejectCounter = ABS(@ARejectCounter - @BRejectCounter)

/*********************************************************************/
/*              Empty Accept Summary Check		             */
/*********************************************************************/
IF @AcceptCounter <> @Zero
BEGIN
	/*************************************************************/
	/*            					             */
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3101 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3102 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3103 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	/*              	KEY				     */
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Gross Shares:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Gross Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KLapseShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3104 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Compensation Income:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Compensation Income:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@KCompIncome,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3105 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Total Taxes:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Total Taxes:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@KTotalTax,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3106 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Total Fees:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Total Fees:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@KFee,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3107 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Tax/Fee Shares:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Tax/Fee Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KTaxShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3108 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3109 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Deferred Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Deferred Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KDefTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3110 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Deferred Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Deferred Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KDefCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3111 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('KEY Total Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('KEY Total Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
				SET @ErrorMsg = '3112 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3113 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	/*              	PSA				     */
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Gross Shares:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Gross Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PLapseShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3201 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Compensation Income:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Compensation Income:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@PCompIncome,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3202 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Total Taxes:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Total Taxes:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@PTotalTax,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3203 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Total Fees:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Total Fees:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@PFee,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3204 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Tax/Fee Shares:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Tax/Fee Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PTaxShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3205 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3206 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Deferred Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Deferred Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PDefTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3207 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Deferred Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Deferred Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PDefCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3208 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('PSA Total Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('PSA Total Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3209 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3210 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	/*              	RSU			             */
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Gross Shares:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Gross Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RLapseShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3301 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Compensation Income:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Compensation Income:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@RCompIncome,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3302 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Total Taxes:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Total Taxes:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@RTotalTax,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3303 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Total Fees:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Total Fees:')) - 40))  + SPACE(1) + LTRIM(CONVERT(Char,@RFee,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3304 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Tax/Fee Shares:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Tax/Fee Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RTaxShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3305 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3306 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Deferred Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Deferred Net Shares:')) - 40))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RDefTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3307 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Deferred Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Deferred Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RDefCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3308 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('RSU Total Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('RSU Total Record(s) Processed:')) - 40)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3309 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
	 	End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3310 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3311 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3312 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	/*              	Combined Totals			     */
	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('Total Record(s) Processed:')) - 55)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@AcceptCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3411 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Gross Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Total Gross Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@TLapseShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3401 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Compensation Income:') + REPLICATE('*',ABS(LEN(RTRIM('Total Compensation Income:')) - 55))  + SPACE(1) + LTRIM(CONVERT(Char,@TCompIncome,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3402 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Taxes:') + REPLICATE('*',ABS(LEN(RTRIM('Total Taxes:')) - 55))  + SPACE(1) + LTRIM(CONVERT(Char,@TotalTax,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3403 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Fees:') + REPLICATE('*',ABS(LEN(RTRIM('Total Fees:')) - 55))  + SPACE(1) + LTRIM(CONVERT(Char,@TFee,1))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3404 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Tax/Fee Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Total Tax/Fee Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@TTaxShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3405 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Total Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Total Net Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber((@PDefTenderShare + @RDefTenderShare + @KDefTenderShare + @TTenderShare))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3410 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Brokerage Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Brokerage Net Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@TTenderShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3406 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Brokerage Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('Brokerage Record(s) Processed:')) - 55)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber((@AcceptCounter - @DEF))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3407 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/		---- 09/01/2008
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Deferred Gross Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Deferred Gross Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@TDefShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3408.1 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/		---- 09/01/2008
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Deferred Gross Tax/Fee Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Deferred Gross Tax/Fee Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@TDefTaxFeeShare)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3408.2 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Deferred Net Shares:') + REPLICATE('*',ABS(LEN(RTRIM('Deferred Net Shares:')) - 55))  + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber((@PDefTenderShare + @RDefTenderShare + @KDefTenderShare))

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3408 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionAcceptReport
	Select	RTRIM('Deferred Record(s) Processed:') + REPLICATE('*',ABS(LEN(RTRIM('Deferred Record(s) Processed:')) - 55)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@DEF)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3409 - tblDistributionAcceptReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End
END
 ELSE
BEGIN
	/*************************************************************/
	/*            					             */
	/*************************************************************/
	Truncate Table tblDistributionAcceptReport

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '3410 - tblDistributionAcceptReport Truncate Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

END

/*********************************************************************/
/*              Empty Reject Summary Check		             */
/*********************************************************************/
IF @RejectCounter <> @Zero
BEGIN
	/*************************************************************/
	/*              	Reject Summary			     */
	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4101 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4102 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	RTRIM('Total Unknown Grant Type Record(s) Rejected:') + REPLICATE('*',ABS(LEN(RTRIM('Total Unknown Grant Type Record(s) Rejected:')) - 50)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@NA)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4103 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	RTRIM('Total KEY Grant Type Record(s) Rejected:') + REPLICATE('*',ABS(LEN(RTRIM('Total KEY Grant Type Record(s) Rejected:')) - 50)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@KEY)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4104 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	RTRIM('Total PSA Grant Type Record(s) Rejected:') + REPLICATE('*',ABS(LEN(RTRIM('Total PSA Grant Type Record(s) Rejected:')) - 50)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@PSA)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4105 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	RTRIM('Total RSU Grant Type Record(s) Rejected:') + REPLICATE('*',ABS(LEN(RTRIM('Total RSU Grant Type Record(s) Rejected:')) - 50)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RSU)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4106 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4107 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	REPLICATE(' ',1)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4108 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

	/*************************************************************/
	Insert	tblDistributionRejectReport
	Select	RTRIM('Total Record(s) Rejected:') + REPLICATE('*',ABS(LEN(RTRIM('Total Record(s) Rejected:')) - 60)) + SPACE(1) + WyethAux.dbo.fnAddCommaToNumber(@RejectCounter)

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4109 - tblDistributionRejectReport Table INSERT Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End

END
 ELSE
BEGIN
	/*************************************************************/
	/*            					             */
	/*************************************************************/
	Truncate Table tblDistributionRejectReport

		Select @ErrorCode = @@ERROR	
		IF @ErrorCode <> @Zero
		 Begin
			SET @ErrorMsg = '4110 - tblDistributionRejectReport Truncate Failure!!!'
			SET @ErrorFlag	= 999
			GOTO Header_Footeer_Error
		 End	

END

/*********************************************************************/
/*              					             */
/*********************************************************************/
Insert	ProcessLog
Values(Getdate(),@Process,@ErrorCode,'SUCCESS','END - WestPort Lapse/Distribution Data From WestPort')

/*********************************************************************/
/*           Commit Transaction				             */
/*********************************************************************/
COMMIT TRAN WestPortRestricted

/*********************************************************************/
/*       Header & Footer Insert Error Logging		             */
/*********************************************************************/
Header_Footeer_Error:

IF @ErrorFlag <> @Zero
 Begin
	ROLLBACK TRAN WestPortRestricted
	Insert	ProcessLog
	Values(Getdate(),@Process,@ErrorCode,'FAILURE',@ErrorMsg)
	RAISERROR (70001,16,1)
	RETURN @ErrorCode
 End

/*********************************************************************/
/*								     */
/*********************************************************************/
EXEC	@ErrorCode = p_WYEtoSSB_restricted

IF @ErrorCode <> @Zero OR @@ERROR <> @Zero
 Begin
	Insert	ProcessLog
	Values(Getdate(),'p_WYEtoSSB_restricted',@ErrorCode,'FAILURE','Wyeth to SSB Restricted Failure')
	RAISERROR (70001,16,2)
	RETURN @ErrorCode
 End

/*********************************************************************/
/*              					             */
/*********************************************************************/
SET NOCOUNT OFF


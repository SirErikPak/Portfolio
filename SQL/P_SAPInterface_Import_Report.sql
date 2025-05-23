USE [SAPInterface]
GO
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[P_SAPInterface_Import_Report]') AND type in (N'P', N'PC'))
DROP PROCEDURE [dbo].[P_SAPInterface_Import_Report]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
Create Proc [dbo].[P_SAPInterface_Import_Report]	(@ErrorCode	Int,
													@Process	Varchar(500))
As
/*********************************************************************/
/*                    TeamConnect SAP Import Update                  */
/*                  						                         */
/* Purpose: SAP Import to TeamConnect Update Report		             */
/*                                                                   */
/*  Org. Date: 01/01/2004        EPAK                                */
/*  Mod. Date: 08/18/2008		 SCHULKD							 */
/*		Changed for TeamConnect tables                               */
/*  Mod. Date: 06/19/2009		 EPAK    							 */
/*       Add Total Amount @ End of the Report						 */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*             Declare Local Variables                               */
/*********************************************************************/
Declare		@Counter		Int,
			@InvoiceKey		Int,
			@CaseName		Varchar(50),
			@CaseNumber		Varchar(20),
			@MatterNumber	Varchar(20),
			@MatterName		Varchar(65),
			@CheckNumber	Varchar(15),
			@TransID		Varchar(16),
			@InvoiceDate	Char(10),
			@InvoiceAmount	Varchar(20),
			@InvoiceNumber	Varchar(50),
			@CheckDate		Char(10),
			@VendorName		Varchar(50),
			@SAPVendorID	Varchar(20),
			@TLength		Int,
			@PLength		Int,
			@InvStatCode	Char(1),
			@Reason			Varchar(1000),
			@TotInvAmt		Money				---- 06/19/2009


Select	@Process = @Process + ' - [P_SAPInterface_Import_Report]',
		@TotInvAmt = 0

/*********************************************************************/
/*  	    SAP Import To TeamConnect Update Process Report          */
/*********************************************************************/
Insert	SAPInterface.dbo.ImportInvoiceReport
Select	CONVERT(Varchar(30), GetDate(), 100)
	+ REPLICATE(' ',40)
	+ 'Updating Invoice From SAP File(s) - Updated List'

	-- Error Check for Insert
	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(1) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End
	-- End of Error Check

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(2) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(3) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	'Matter Number' 
	+ REPLICATE(' ',14)  + 'Matter Name' 
	+ REPLICATE(' ',59) + 'Vendor'
	+ REPLICATE(' ',30) + 'SAP Vendor ID' 
	+ REPLICATE(' ',10) + 'Invoice Number' 
	+ REPLICATE(' ',5)  + 'Invoice Date' 
	+ REPLICATE(' ',5)  + 'Payment Amount (USD)'
	+ REPLICATE(' ',8)  + 'Check Number'
	+ REPLICATE(' ',9)  + 'Check Date' 

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(4) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(5) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(6) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

/*********************************************************************/
/************ Updated SAP Invoice to TeamConnect Report **************/
/*********************************************************************/
DECLARE ReceivedReport_CURSOR CURSOR FOR
	Select	INV.Invoice_Key,
			EXT.RemitMessage,
			INV.Matter_Name,
			INV.Matter_Number,
			EXT.VendorName,
			EXT.SAPVendor,
			EXT.InvoiceNumber,
			InvoiceDate =
				CASE
				 WHEN EXT.InvoiceDate = ''  THEN 'UNKNOWN'
				 WHEN EXT.InvoiceDate = '0' THEN 'UNKNOWN'
				 WHEN EXT.InvoiceDate IS NULL THEN 'UNKNOWN'
				 ELSE Substring(EXT.InvoiceDate,5,2) + '/' + Substring(EXT.InvoiceDate,7,2) + + '/' + Substring(EXT.InvoiceDate,1,4)
				END,
			InvoiceAmount =
				CASE
				 WHEN EXT.InvoiceAmountPaid IS NULL THEN 'UNKNOWN'
				 WHEN EXT.InvoiceAmountPaid = '' THEN 'UNKNOWN'
				 ELSE '$' + LTRIM(Convert(Char,(Convert(Money,EXT.InvoiceAmountPaid)),1))
				END,
			EXT.CheckNumber,
			InvoiceCheckDate =
			CASE
			 WHEN EXT.PayDate = ''  THEN 'UNKNOWN'
			 WHEN EXT.PayDate = '0' THEN 'UNKNOWN'
			 WHEN EXT.PayDate IS NULL THEN 'UNKNOWN'
			 ELSE Substring(EXT.PayDate,5,2) + '/' + Substring(EXT.PayDate,7,2) + + '/' + Substring(EXT.PayDate,1,4)
			END

	From	SAPInterface.dbo.ImportInvoice EXT 
			INNER JOIN tc_wyeth.dbo.SAPSentForPayment INV
			ON EXT.SAPVendor = INV.SAP_VEND_ID
	AND		Substring(INV.Invoice_Number,1,16) = RTRIM(EXT.InvoiceNumber)
	AND		REPLICATE('0', 4 - DATALENGTH(RTRIM(Convert(Char(4),DATEPART(Year,INV.Invoice_Date))))) + RTRIM(Convert(Char(4),DATEPART(Year,INV.Invoice_Date))) +
			REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Month,INV.Invoice_Date))))) + RTRIM(Convert(Char(2),DATEPART(Month,INV.Invoice_Date))) +
			REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Day,INV.Invoice_Date))))) + RTRIM(Convert(Char(2),DATEPART(Day,INV.Invoice_Date))) = EXT.InvoiceDate
	Where	EXT.Processed IS NULL

/*********************************************************************/
/*                   OPEN Cursor                                     */ 
/*********************************************************************/
OPEN ReceivedReport_CURSOR

FETCH NEXT FROM ReceivedReport_CURSOR INTO	@InvoiceKey,
											@CaseName,
											@MatterName,
											@MatterNumber,
											@VendorName,
											@SAPVendorID,
											@InvoiceNumber,
											@InvoiceDate,
											@InvoiceAmount,
											@CheckNumber,
											@CheckDate

WHILE (@@FETCH_STATUS <> -1)
  BEGIN
	IF (@@FETCH_STATUS <> -2)
	BEGIN

	INSERT	SAPInterface.dbo.ImportInvoiceReport
	SELECT	RTRIM(@MatterNumber)  + REPLICATE(' ',ABS(LEN(RTRIM(@MatterNumber)) - 27))
			+ RTRIM(@MatterName)  + REPLICATE(' ',ABS(LEN(RTRIM(@MatterName)) - 70))
			+ RTRIM(@VendorName)  + REPLICATE(' ',ABS(LEN(RTRIM(@VendorName)) - 40))
			+ RTRIM(@SAPVendorID) + REPLICATE(' ',ABS(LEN(RTRIM(@SAPVendorID)) - 20))
			+ RTRIM(@InvoiceNumber) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceNumber)) - 20))
			+ RTRIM(@InvoiceDate) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceDate)) - 20))
			+ RTRIM(@InvoiceAmount) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceAmount)) - 25))
			+ RTRIM(@CheckNumber) + REPLICATE(' ',ABS(LEN(RTRIM(@CheckNumber)) - 20))
			+ RTRIM(@CheckDate) + REPLICATE(' ',ABS(LEN(RTRIM(@CheckDate)) - 20))

		SELECT	@ErrorCode = @@Error
		IF	@ErrorCode <> 0

		BEGIN
			INSERT SAPInterface.dbo.ProcessLog
			Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'ReceivedReport Cursor Failed')
			CLOSE ReceivedReport_CURSOR
			DEALLOCATE ReceivedReport_CURSOR
			RETURN @ErrorCode
		END

	 END

	FETCH NEXT FROM ReceivedReport_CURSOR INTO	@InvoiceKey,
												@CaseName,
												@MatterName,
												@MatterNumber,
												@VendorName,
												@SAPVendorID,
												@InvoiceNumber,
												@InvoiceDate,
												@InvoiceAmount,
												@CheckNumber,
												@CheckDate
	END


/*********************************************************************/
/*             Close & Deallocate Cursor							 */
/*********************************************************************/
CLOSE ReceivedReport_CURSOR
DEALLOCATE ReceivedReport_CURSOR

/*********************************************************************/
/*                   End Of Update Report                            */ 
/*********************************************************************/
Select	@Counter = Count(*)
From	SAPInterface.dbo.ImportInvoice
Where	Processed IS NULL

---- 06/19/2009
Select	@TotInvAmt = SUM(CONVERT(Money,InvoiceAmountPaid))
From	SAPInterface.dbo.ImportInvoice
Where	Processed IS NULL

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(7) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(8) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	'** Total: ' + RTRIM(Convert(Char,@Counter)) + ' Invoice(s)'

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(9) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

---- 06/19/2009
Insert	SAPInterface.dbo.ImportInvoiceReport
Select	'** Total Invoice Amount: ' + '$' + LTRIM(Convert(Char,(Convert(Money,@TotInvAmt)),1))

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(9.1) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(10) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportInvoiceReport
Select	REPLICATE('*',75) + ' End of Report ' + REPLICATE('*',74)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(11) SAPInterface.dbo.ImportInvoiceReport Failed')
		RETURN @ErrorCode
	End

/*********************************************************************/
/*  	    SAP Import To LawManager Rejected Process Report         */
/*********************************************************************/
Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	CONVERT(Varchar(30), GetDate(), 100)
	+ REPLICATE(' ',40)
	+ 'Rejected Invoice From SAP File(s) - Error List'

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(1) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(2) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(3) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	'Matter Number' 
	+ REPLICATE(' ',14)  + 'Matter Name' 
	+ REPLICATE(' ',59) + 'Vendor'
	+ REPLICATE(' ',30) + 'SAP Vendor ID' 
	+ REPLICATE(' ',10) + 'Invoice Number' 
	+ REPLICATE(' ',5)  + 'Invoice Date' 
	+ REPLICATE(' ',5)  + 'Payment Amount (USD)'
	+ REPLICATE(' ',8)  + 'Check Number'
	+ REPLICATE(' ',9)  + 'Check Date'
	+ REPLICATE(' ',11)  + 'Reason'
		

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(4) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(5) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(6) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

/*********************************************************************/
/************ Rejected SAP Invoice to LawManager Report **************/
/*********************************************************************/
DECLARE RejectedReport_CURSOR CURSOR FOR
	Select	INV.Invoice_Key,
			EXT.RemitMessage,
			ISNULL(INV.Matter_Name,'UNKNOWN'),
			ISNULL(INV.Matter_Number,'UNKNOWN'),
			ISNULL(EXT.VendorName,'UNKNOWN'),
			ISNULL(EXT.SAPVendor,'UNKNOWN'),
			ISNULL(EXT.InvoiceNumber,'UNKNOWN'),
			InvoiceDate =
				CASE
				 WHEN EXT.InvoiceDate = ''  THEN 'UNKNOWN'
				 WHEN EXT.InvoiceDate = '0' THEN 'UNKNOWN'
				 WHEN EXT.InvoiceDate IS NULL THEN 'UNKNOWN'
				 ELSE Substring(EXT.InvoiceDate,5,2) + '/' + Substring(EXT.InvoiceDate,7,2) + + '/' + Substring(EXT.InvoiceDate,1,4)
				END,
			InvoiceAmount =
				CASE
				 WHEN EXT.InvoiceAmountPaid IS NULL THEN 'UNKNOWN'
				 WHEN EXT.InvoiceAmountPaid = '' THEN 'UNKNOWN'
				 ELSE '$' + LTRIM(Convert(Char,(Convert(Money,EXT.InvoiceAmountPaid)),1))
				END,
			EXT.CheckNumber,
			InvoiceCheckDate =
			CASE
			 WHEN EXT.PayDate = ''  THEN 'UNKNOWN'
			 WHEN EXT.PayDate = '0' THEN 'UNKNOWN'
			 WHEN EXT.PayDate IS NULL THEN 'UNKNOWN'
			 ELSE Substring(EXT.PayDate,5,2) + '/' + Substring(EXT.PayDate,7,2) + + '/' + Substring(EXT.PayDate,1,4)
			END,
			IER.ErrorMsg

	From	SAPInterface.dbo.ImportInvoice EXT 
			LEFT JOIN tc_wyeth.dbo.SAPSentForPayment INV
			ON EXT.SAPVendor = INV.SAP_VEND_ID
	AND		Substring(INV.Invoice_Number,1,16) = RTRIM(EXT.InvoiceNumber)
	AND		REPLICATE('0', 4 - DATALENGTH(RTRIM(Convert(Char(4),DATEPART(Year,INV.Invoice_Date))))) + RTRIM(Convert(Char(4),DATEPART(Year,INV.Invoice_Date))) +
			REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Month,INV.Invoice_Date))))) + RTRIM(Convert(Char(2),DATEPART(Month,INV.Invoice_Date))) +
			REPLICATE('0', 2 - DATALENGTH(RTRIM(Convert(Char(2),DATEPART(Day,INV.Invoice_Date))))) + RTRIM(Convert(Char(2),DATEPART(Day,INV.Invoice_Date))) = EXT.InvoiceDate
			LEFT JOIN ImportErrorRecord IER
			ON IER.InvoiceNumber = RTRIM(EXT.InvoiceNumber)

	Where	EXT.Processed IS NOT NULL

/*********************************************************************/
/*                   OPEN Cursor                                     */ 
/*********************************************************************/
OPEN RejectedReport_CURSOR

	FETCH NEXT FROM RejectedReport_CURSOR INTO	@InvoiceKey,
												@CaseName,
												@MatterName,
												@MatterNumber,
												@VendorName,
												@SAPVendorID,
												@InvoiceNumber,
												@InvoiceDate,
												@InvoiceAmount,
												@CheckNumber,
												@CheckDate,
												@Reason

/*********************************************************************/
/*               Concat Files together   		                     */
/*         Insert INTO SAPInterface.dbo.ExtractOut                   */
/*********************************************************************/
WHILE (@@FETCH_STATUS <> -1)
  BEGIN
	IF (@@FETCH_STATUS <> -2)
	BEGIN

	INSERT	SAPInterface.dbo.ImportRejectedInvoiceReport
	SELECT	RTRIM(@MatterNumber)  + REPLICATE(' ',ABS(LEN(RTRIM(@MatterNumber)) - 27))
			+ RTRIM(@MatterName)  + REPLICATE(' ',ABS(LEN(RTRIM(@MatterName)) - 70))
			+ RTRIM(@VendorName)  + REPLICATE(' ',ABS(LEN(RTRIM(@VendorName)) - 40))
			+ RTRIM(@SAPVendorID) + REPLICATE(' ',ABS(LEN(RTRIM(@SAPVendorID)) - 20))
			+ RTRIM(@InvoiceNumber) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceNumber)) - 20))
			+ RTRIM(@InvoiceDate) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceDate)) - 20))
			+ RTRIM(@InvoiceAmount) + REPLICATE(' ',ABS(LEN(RTRIM(@InvoiceAmount)) - 25))
			+ RTRIM(@CheckNumber) + REPLICATE(' ',ABS(LEN(RTRIM(@CheckNumber)) - 20))
			+ RTRIM(@CheckDate) + REPLICATE(' ',ABS(LEN(RTRIM(@CheckDate)) - 20))
		    + LTRIM(RTRIM(@Reason))

		Select	@ErrorCode = @@Error
		IF @ErrorCode <> 0

		Begin
			INSERT SAPInterface.dbo.ProcessLog
			Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'RejectedReport Cursor Failed')
			CLOSE RejectedReport_CURSOR
			DEALLOCATE RejectedReport_CURSOR
			RETURN @ErrorCode
		End

      End

	FETCH NEXT FROM RejectedReport_CURSOR INTO	@InvoiceKey,
												@CaseName,
												@MatterName,
												@MatterNumber,
												@VendorName,
												@SAPVendorID,
												@InvoiceNumber,
												@InvoiceDate,
												@InvoiceAmount,
												@CheckNumber,
												@CheckDate,
												@Reason

  END

/*********************************************************************/
/*             Close & Deallocate Cursor					         */
/*********************************************************************/
CLOSE RejectedReport_CURSOR
DEALLOCATE RejectedReport_CURSOR

/*********************************************************************/
/*                   End Of Rejected Report                          */ 
/*********************************************************************/
Insert	ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', '(6.1) Insert RejectedReport Description Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	'--> No Entry in INVOICE Table OR Status is NOT Paid'

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', '(6.2) Insert RejectedReport Description Failed')
		RETURN @ErrorCode
	End

Select	@Counter = Count(*)
From	SAPInterface.dbo.ImportInvoice
Where	Processed IS NOT NULL

Insert	ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(7) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(8) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	'** Total: ' + RTRIM(Convert(Char,@Counter)) + ' Invoice(s)'

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(9) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE(' ',1)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(10) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

Insert	SAPInterface.dbo.ImportRejectedInvoiceReport
Select	REPLICATE('*',75) + ' End of Report ' + REPLICATE('*',74)

	Select	@ErrorCode = @@Error
	IF @ErrorCode <> 0

	Begin
		INSERT SAPInterface.dbo.ProcessLog
		Values(Getdate(), @Process, @ErrorCode, 'WARNING', 'Insert(11) SAPInterface.dbo.ImportRejectedInvoiceReport Failed')
		RETURN @ErrorCode
	End

/*********************************************************************/
/*																     */
/*********************************************************************/
SET NOCOUNT OFF
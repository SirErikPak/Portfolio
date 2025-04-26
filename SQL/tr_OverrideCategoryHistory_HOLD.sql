IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_OverrideCategoryHistory]'))
DROP TRIGGER [dbo].[tr_OverrideCategoryHistory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_OverrideCategoryHistory] ON [dbo].[OverrideCategoryHistory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iEmpId			Varchar(10),
			@iCategoryCode	Varchar(10),
			@iCategoryValue	Varchar(25),
			@iLastUser		Int,
			@iLastUpdate	Char(10),
			@iArchiveDate	Char(10),
			@cEmpId			Varchar(10),
			@cCategoryCode	Varchar(10),
			@cCategoryValue	Varchar(25),
			@cLastUser		Int,
			@cLastUpdate	Char(10),
			@cArchiveDate	Char(10)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(OC.EmpId) 
	From OverrideCategoryHistory OC INNER JOIN deleted DEL 
	ON OC.EmpId = DEL.EmpId
	AND OC.CategoryCode = DEL.CategoryCode) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo			Int		Identity,
			EmpID			Varchar(10)	NOT NULL,
			CategoryCode	Varchar(10) NOT NULL
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	EmpID,
				CategoryCode
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(EmpID),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP		
		
			/*********************************************************/
			/*                                                       */
			/*********************************************************/		
			Select	@iEmpId			= D.EmpId,
					@iCategoryCode	= D.CategoryCode,
					@iCategoryValue	= D.CategoryValue,
					@iLastUser		= D.LastUser,
					@iLastUpdate	= D.LastUpdate,
					@iArchiveDate	= D.ArchiveDate
			From	OverrideCategoryHistory O INNER JOIN deleted D
					ON O.EmpId = D.EmpId
					AND O.CategoryCode = D.CategoryCode		
			Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/	
			Select	@cEmpId			= O.EmpId,
					@cCategoryCode	= O.CategoryCode,
					@cCategoryValue	= O.CategoryValue,
					@cLastUser		= O.LastUser,
					@cLastUpdate	= O.LastUpdate,
					@cArchiveDate	= O.ArchiveDate
			From	OverrideCategoryHistory O INNER JOIN deleted D
					ON O.EmpId = D.EmpId
					AND O.CategoryCode = D.CategoryCode
			Where	O.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			And		O.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			IF	(
				-----(@iEmpId =@cEmpId) And (@iCategoryCode = @cCategoryCode) And 
				(@iCategoryValue = @cCategoryValue) And
				(@iLastUser = @cLastUser) And -----(@iLastUpdate = @cLastUpdate) And 
				(@iArchiveDate = @cArchiveDate)
				)
			 Begin
				/*****************************************************/
				/*					No Change(s)                     */
				/*****************************************************/
				GOTO NextRecordCheck
					 
			 End
			Else
			 Begin
				/*****************************************************/
				/*				Update Change(s)                     */
				/*****************************************************/
				Insert	AuditOverrideCategoryHistory
				SELECT	D.EmpId,
						D.CategoryCode,
						D.CategoryValue,
						D.LastUser,
						D.LastUpdate,
						D.ArchiveDate,
						@Info,
						GETDATE(),
						USER_NAME()
				From	OverrideCategoryHistory O INNER JOIN deleted D
						ON O.EmpId = D.EmpId
						AND O.CategoryCode = D.CategoryCode
				Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
				And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
								 
		 End

				/*****************************************************/
				/*                                                   */
				/*****************************************************/
				NextRecordCheck:	
				Select	@SeqNo = @SeqNo + @Min

			 End		-- END WHILE LOOP
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Drop Table #Temp
									
	End
 Else
	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditOverrideCategoryHistory
		SELECT	EmpId,
				CategoryCode,
				CategoryValue,
				LastUser,
				LastUpdate,
				ArchiveDate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted

	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO